import glob
import itertools
import json
import os
from abc import ABC, abstractmethod
from multiprocessing import Manager, Process
from shutil import rmtree
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, layers, losses, metrics, models, utils
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
    DPKerasAdamOptimizer,
)
from util import figures as figure_helper
from util import metrics as metric_helper
from util import models as model_helper
from util import utilities as util

from core.base import BaseClass
from core.dataset import (
    BaseImageClass,
    BaseTimeSeriesClass,
    CelebAConditionalContainer,
    CelebAContainer,
    DataContainer,
    LFWContainer,
    MotionSenseConditionalContainer,
    PictureContainer,
    SimpleConditionalDataGenerator,
    SimpleDataGenerator,
)

"""Log Levels ('verbose' parameter)
0 -- silent
1 -- minimal log output
2 -- detailed log output

"""


class ModelContainer(BaseClass, ABC):
    """Baseclass for models. Determines shared logic for model handling."""

    _configs_dir: str = "configs/model_configs"
    _models_dir: str = "models"

    _supported_subclasses: dict = {
        "GAN": "GANContainer",
        "BasicPictureVAE": "BasicPictureVAE",
        "Cifar10VAE": "Cifar10VAE",
        "PlainPictureVAE": "PlainPictureVAE",
        "ConditionalPlainPictureVAE": "ConditionalPlainPictureVAE",
        "DFCPictureVAE": "DFCPictureVAE",
        "MultitaskPlainPictureVAE": "MultitaskPlainPictureVAE",
        "MultitaskBinVAE": "MultitaskBinVAE",
        "MultitaskSensorDataVAE": "MultitaskSensorDataVAE",
    }

    _supported_datacontainer: dict = {
        "lfw": LFWContainer,
        "celeba": CelebAContainer,
        "mnist": PictureContainer,
        "cifar10": PictureContainer,
        "fashion_mnist": PictureContainer,
        "celebaconditional": CelebAConditionalContainer,
        "motionsenseconditional": MotionSenseConditionalContainer,
    }

    def __init__(self, config_name: str, config: dict, verbose: int = 1) -> None:
        """Initialize for model container. Do not call directly!
            Use `ModelContainer.create` to get instances of the actual model classes

        Args:
            config_name (str): Name of the config file to use.
            config (dict): Loaded content of the config file.
            verbose (int, optional): Verbosity level used for logging. Defaults to 1.

        Raises:
            Error: Supplied config not valid.
        """

        self._verbose = verbose

        self._trained_: bool = False
        self._dim_y = None
        self._train_history: dict = None

        self.data: DataContainer = None
        self._data_indices: np.array = None
        self._data_hash: str = None
        self._data_shape: Sequence[int] = None
        self._checkpoint_mode: bool or int = False
        self._checkpoint_format_string: str = "weights-{epoch:02d}.hdf5"

        # Set base path
        self._base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Create a helper instance which can be used at any point
        self._util = util.BaseHelper(self)

        # Log versions
        self._util.log(
            f"{self.__class__} starts with \n\tKeras Version: {tf.keras.__version__}\n\tTensorflow Version: {tf.__version__}",
            2,
        )

        # Validate the config
        reason = self._util.config_valid(config)

        if reason is not None:
            raise ValueError(
                f"{self.__class__} supplied config not valid due to reason: {reason}"
            )

        self._model_name = config_name
        self.config_name_ = config_name
        self._config = config

        self.data_cls = ModelContainer._supported_datacontainer[
            config["dataset"].split("-")[0].lower()
        ]

    @staticmethod
    def create(config_name: str, verbose: int = 1) -> "ModelContainer":
        """Create a model container due to a supplied config.

        Args:
            config_name (str): Name of the config file to use.
            verbose (int, optional): Verbosity level used for logging. Defaults to 1.

        Raises:
            NotImplementedError: Requested model type not present.

        Returns:
            ModelContainer: Instance of the requested model
        """

        # Set base path to be one folder below current directory
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if config_name.endswith(".json"):
            config_name = config_name[:-5]

        # Load model config file
        config_file = os.path.join(
            base_path, ModelContainer._configs_dir, config_name + ".json"
        )
        with open(config_file, "r") as f:
            config = json.loads(f.read())

        model_type = util.BaseHelper._read_config(config, "type", strict=True)

        if model_type not in ModelContainer._supported_subclasses.keys():
            raise NotImplementedError(f'Unknown model type "{model_type}"')

        return globals()[ModelContainer._supported_subclasses[model_type]](
            config_name, config, verbose
        )

    def _save_model(self, models: list) -> None:
        """Base function for model saving

        Is used by all subclasses in specific save functions.
        Saves weight files, data permutation, training history and a data hash.

        Args:
            models (list): List containing one entry per (sub)model, with the model object and a name to use for the weight file.
            save_samples (bool, optional): If True the data permutation that was used to shuffle the data is saved. Defaults to True.
        """

        # Create model folder
        model_dir = self.get_model_dir()

        # Save all models
        for entry in models:
            entry["model"].save_weights(
                os.path.join(model_dir, entry["name"] + "_weights.tf"),
            )

        self._util.log(f"{self.__class__} saved {models} to `{model_dir}`", 1)

        # Only data permutation has to be saved
        if self._data_indices is not None:
            np.savez(os.path.join(model_dir, "data_indices.npz"), **self._data_indices)
            self._util.log(f"{self.__class__} saved data indices to `{model_dir}`", 2)

        # Save training history (losses)
        # TODO save model history data in a different way
        if self._train_history:
            hist_df = pd.DataFrame(self._train_history)  # required
            with open(os.path.join(model_dir, "train_history.json"), "w") as f:
                hist_df.to_json(f)
            self._util.log(f"{self.__class__} saved model history to `{model_dir}`", 2)

        # Save data hash
        if self._data_hash:
            with open(os.path.join(model_dir, "data_hash.txt"), "w") as f:
                f.write("{0}".format(self._data_hash))

            self._util.log(f"{self.__class__} saved data hash {self._data_hash}.", 2)

    def _load_model(self, model_names: Sequence[str]) -> List["models.Model"]:
        """Base function for model loading

        Is used by all subclasses in specific load functions.
        Imports weight files from model directory.

        Args:
            model_names (Sequence[str]): The names of the weight files to import.

        Raises:
            Exception: If the model directory is not present.
            Exception: If a model weight file is not present.

        Returns:
            List["models.Model"]: A list of tensorflow models with imported weigths.
        """

        model_dir = self.get_model_dir()

        # Load all models
        models = list()

        # Model needs to be created first
        # (cannot be loaded because of lambda layer)
        self.create_model(trained=True)

        for name in model_names:
            model = getattr(self, name)

            weight_file = os.path.join(model_dir, name + "_weights")
            if os.path.isfile(weight_file + ".tf") or os.path.isfile(
                weight_file + ".tf.index"
            ):
                model.load_weights(weight_file + ".tf")
            elif os.path.isfile(weight_file + ".h5"):
                model.load_weights(weight_file + ".h5", by_name=True)
            else:
                raise Exception(
                    f"Model weight file `{weight_file} (.tf/.h5)` not found. Make sure the model was already trained.\n\tTo create a new model use `load_data` and afterwards `create_model`."
                )

            models.append(model)

        self._util.log("Model successfully loaded", required_level=1)

        return models

    def clear_model(self) -> None:
        """Deletes the model directory and all its content."""
        model_dir = self.get_model_dir()
        rmtree(model_dir)

    def load_data(self) -> None:
        """Load model training/test/validation data

        Uses dataset module methods to acquire and preprocess data.
        Also stores the sets within instance variables.

        Raises:
            Exception: If size of noise is not larger or equal compared to the amount of labels when trying to create a GAN model.
        """

        data_dir, dataset = self._util.read_config("data_dir", "dataset")

        data_conf = self._util.get_partial_config_dict(
            "stratify_split",
            "train_size",
            "val_size",
            "force_new_data_indices",
            "shuffle_data",
            "train",
            "test",
            "val",
            strict=False,
        )

        data_conf["only_classes"] = self._util.read_config("use_classes", strict=False)

        if self._util.read_config("per_example_loss", strict=False):
            data_conf["align_with_batch_size"] = self._util.read_config("batch_size")
        else:
            data_conf["align_with_batch_size"] = False

        perturbation_conf = self._util.get_partial_config_dict(
            "data_ldp_noise", "epsilon", "m", "b", strict=False
        )

        self._data_indices = None

        path_to_data_indices = os.path.join(self.get_model_dir(), "data_indices.npz")

        if os.path.isfile(path_to_data_indices):
            self._data_indices = np.load(path_to_data_indices, allow_pickle=True)
            self._util.log(
                f"{self.__class__} loaded data indices from `{path_to_data_indices}`", 2
            )

        self.data = self.data_cls(
            self._util,
            os.path.join(self._base_path, data_dir),
            dataset,
            data_conf,
            perturbation_conf,
            self._data_indices,
        )

        # Create hash to be able to compare data
        self._data_hash = self.data.get_hash()
        self._data_indices = self.data.get_data_indices()

        if self._type == "GAN":

            # ATTENTION: Concatenates z (noise) and y (condition) -> dim_y has to be the same as dim_z
            dim_z = self._util.read_config("dim_z")
            self._dim_y = dim_z

            # dim_z has to be larger than the amount of labels we have (as dim_y is set to the same value and to
            # ensure one-hot encoding)
            num_labels = self.data.get_num_unique_labels()
            if dim_z < num_labels:
                raise ValueError(
                    f"{self.__class__} dim_z (noise size) from config must be larger or equal compared to the amount of labels (but is only {dim_z} vs. {num_labels})"
                )

        # Load and preprocess data
        self.data.preprocess_data(self._type, self._dim_y)

        # Log data details
        # TODO check naming convention!
        self._util.log(
            f"{self.__class__} uses \n\tTraining Samples: {len(self.data.X_train)}\n\tValidation Samples: {len(self.data.X_val)}\n\tTest Samples: {len(self.data.X_test)}",
            2,
        )

        # Determine data shape (without sample size)
        # assume train test and val have the same shape
        self._data_shape = self.data.get_data_shape()[0][1:]

    def check_hash(self) -> None:
        """Checks the hash of the loaded data against a saved hash.

        Raises:
            FileNotFoundError: File with old hash not present.
            AssertionError: Hashes don't match
        """
        if not self._data_hash:
            self._util.log(
                f"{self.__class__} misses new data hash for check. Did you load data?",
                1,
            )
            return

        model_dir = self.get_model_dir()
        old_hash_path = os.path.join(model_dir, "data_hash.txt")

        if not os.path.isfile(old_hash_path):
            raise FileNotFoundError(
                f"{self.__class__} misses old hash at `{old_hash_path}`"
            )

        with open(old_hash_path, "r") as fr:
            old_text_hash = fr.read()

        if not old_text_hash == (new_data_hash := self._data_hash):
            raise AssertionError(
                f"Hashes do not match for {self.__class__}. Its `{new_data_hash}` but should be `{old_text_hash}`."
            )

        self._util.log(f"{self.__class__} successfully compared data hash.", 1)

    def get_model_dir(self) -> str:
        """Returns the path to the directory all model information is stored in.
        We create a hierarchy that allows for automatic evaluation more easily.
        When the folder does not exist, it will be created.

        Returns:
            str: String representation of absolute path to the model directory.
        """

        (
            dataset,
            model_type,
            ldp_lower_bound,
            data_ldp_noise,
            epsilon,
            opt_param,
        ) = self._util.read_config(
            "dataset",
            "type",
            "ldp_lower_bound",
            "data_ldp_noise",
            "epsilon",
            "optimizer_params",
            strict=False,
        )

        main_path = os.path.join(self._base_path, self._models_dir, dataset, model_type)

        if (ldp_lower_bound is not None) and (ldp_lower_bound > 0):
            model_dir = os.path.join(
                main_path,
                "VAE-LDP-gen",
                str(ldp_lower_bound),
            )
        elif (data_ldp_noise is not None) and (data_ldp_noise > 0):
            model_dir = os.path.join(
                main_path,
                "VAE-LDP",
                str(data_ldp_noise),
            )
        elif (epsilon is not None) and (epsilon > 0):
            model_dir = os.path.join(
                main_path,
                "LDP",
                str(epsilon),
            )
        elif "noise_multiplier" in opt_param.keys():
            model_dir = os.path.join(
                main_path,
                "CDP",
                str(opt_param["noise_multiplier"]),
            )
        else:
            model_dir = os.path.join(main_path, "orig")

        full_path = os.path.join(model_dir, self._model_name)

        if not os.path.isdir(full_path):
            os.makedirs(full_path)

        return full_path

    def get_figure_dir(self) -> str:
        """Returns the path to the figures directory, where all related figures should be stored. Creates the dir if necessary.

        Returns:
            str: Absolute path to the figure directory
        """
        if chkpt := self._checkpoint_mode:
            figure_dir = os.path.join(
                self.get_model_dir(), "checkpoint_results", str(chkpt), "figures"
            )
        else:
            figure_dir = os.path.join(self.get_model_dir(), "figures")

        if not os.path.isdir(figure_dir):
            os.makedirs(figure_dir)

        return figure_dir

    def get_attack_results_dir(self) -> str:
        """Returns the path to the attack results directory, where attack_results.json should be stored. Creates the dir if necessary.

        Returns:
            str: Absolute path to the attack results directory
        """
        if chkpt := self._checkpoint_mode:
            attack_results_dir = os.path.join(
                self.get_model_dir(), "checkpoint_results", str(chkpt)
            )
        else:
            attack_results_dir = self.get_model_dir()

        if not os.path.isdir(attack_results_dir):
            os.makedirs(attack_results_dir)

        return attack_results_dir

    def get_checkpoint_dir(self) -> str:
        """Returns the path to the checkpoints directory, where all checkpoints should be stored. Creates the dir if necessary.

        Returns:
            str: Absolute path to the figure directory
        """
        chkpt_dir = os.path.join(self.get_model_dir(), "checkpoints")

        if not os.path.isdir(chkpt_dir):
            os.makedirs(chkpt_dir)

        return chkpt_dir

    def get_all_checkpoints(self, epochs_only: bool = True) -> list:
        """Scans the checkpoint dir and returns a list of all checkpoints or the corresponding epochs.

        Args:
            epochs_only_bool (bool, optional): Whether to return only the epochs. Defaults to True.

        Returns:
            list: List of checkpoints or epochs
        """

        checkpoint_dir = self.get_checkpoint_dir()
        format_string, extension = os.path.splitext(self._checkpoint_format_string)

        format_string = format_string.split("-")[0] + "-"

        filenames = glob.iglob(
            os.path.join(checkpoint_dir, format_string + "[0-9]*" + extension)
        )

        list_of_chkpts = [os.path.basename(f) for f in filenames]

        if epochs_only:
            list_of_chkpts = [
                int(f.replace(format_string, "").replace(extension, ""))
                for f in list_of_chkpts
            ]

        return list_of_chkpts

    def check_if_data_shape_is_present(self) -> None:
        """Checks if data shape was set.

        Raises:
            RuntimeError: Data shape is not present.
        """

        if self._data_shape is None:
            raise RuntimeError(
                f"{self.__class__} couldn't determine data shape, load data first."
            )

    def set_optimizer(self) -> None:
        opt_name = self._util.read_config("optimizer", strict=True)
        opt_param = self._util.read_config("optimizer_params", strict=True)
        if opt_name not in globals():
            raise NotImplementedError(f"Optimizer {opt_name} not Implemented")
        self.optimizer = globals()[opt_name](**opt_param)

    def get_per_example_loss(self) -> bool:
        return self._util.read_config("per_example_loss", strict=False)

    def get_model_checkpoint(self) -> int:
        return self._util.read_config("checkpoint_model", strict=False)

    def save_evaluation_metric(self, name: str, results: dict) -> None:

        if chkpt := self._checkpoint_mode:
            save_dir = os.path.join(
                self.get_model_dir(),
                "checkpoint_results",
                str(chkpt),
            )
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            save_file = os.path.join(save_dir, "evaluation_results.json")
        else:
            save_file = os.path.join(self.get_model_dir(), "evaluation_results.json")

        if os.path.isfile(save_file):
            with open(save_file, "r") as f:
                prev_results = json.load(f)
        else:
            prev_results = dict()

        prev_results[name] = results

        with open(save_file, "w") as f:
            json.dump(prev_results, f, cls=util.CustomJSONEncoder)

    def plot_model_history(self) -> None:
        """Loads model history file if no history is present, plots it and saves it to the figures directory.

        Raises:
            FileNotFoundError: When no history is present and history file is missing.
        """

        if not self._train_history:
            history_file = os.path.join(self.get_model_dir(), "train_history.json")

            if not os.path.isfile(history_file):
                raise FileNotFoundError(
                    f"{self.__class__} misses train_history.json. Run training first and save model."
                )

            with open(history_file, "r") as fr:
                hist = json.load(fr)
        else:
            hist = self._train_history

        figure_helper.plot_values_over_keys(
            data=hist,
            savepath=os.path.join(self.get_figure_dir(), "training_history.pdf"),
        )

    def flush_config(self) -> None:
        """Writes the currently loaded config and any changes to the config file."""

        config_file = os.path.join(
            self._base_path, ModelContainer._configs_dir, self.config_name_ + ".json"
        )

        with open(config_file, "w") as fw:
            json.dump(self._config, fw, cls=util.CustomJSONEncoder)

    def compute_epsilon_delta_with_rdp_accountant(self):

        epochs, batch_size, num_train_samples = self._util.read_config(
            "epochs", "batch_size", "train_size"
        )

        if num_train_samples < 1:
            self.check_if_data_shape_is_present()
            num_train_samples = self.data.get_data_shape()[0][0]

        opt_params = self._util.read_config("optimizer_params", strict=False)

        if not "noise_multiplier" in opt_params:
            return
        else:
            noise_multiplier = opt_params["noise_multiplier"]

        epsilon, delta = model_helper.compute_epsilon_delta_with_rdp(
            epochs, batch_size, noise_multiplier, num_train_samples
        )

        self._config["cdp_epsilon"] = epsilon
        self._config["cdp_delta"] = delta

        self.flush_config()

    """
    Methods to be defined in the subclasses for the concrete model types
    """

    @abstractmethod
    def create_model(self, trained: bool):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_checkpoint_at(self, epoch: int):
        pass


# Generative Adversarial Network
class GANContainer(ModelContainer):
    """GAN specific container class"""

    # GAN specific attributes
    generator: "models.Model" = None
    discriminator: "models.Model" = None
    gan: "models.Model" = None

    def __init__(self, model_config: str, config: dict, verbose: int = 1):
        """GAN Constructor

        Do not call directly. Use ModelContainer.create to construct instances.

        Args:
            model_config (str): Name of the model config.
            verbose (int, optional): Verbosity level. Is used for logging. Defaults to 1.
        """
        self._type = "GAN"
        super().__init__(model_config, config, verbose)

    def create_model(self, trained: bool = False):
        """Create GAN model

        Args:
            trained (bool, optional): Indicates if model was already trained before.
                Not used in GAN subclass. Defaults to False.

        Raises:
            Exception: Raised when data is not present and data shape cannot be determined.

        """

        self.check_if_data_shape_is_present()

        g = self._generator_model()
        g.summary(print_fn=lambda x: self._util.log(x, 2))

        d = self._discriminator_model()
        d.summary(print_fn=lambda x: self._util.log(x, 2))

        gd = self._generator_containing_discriminator(g, d)
        gd.summary(print_fn=lambda x: self._util.log(x, 2))

        self.set_optimizer()

        g.compile(loss="binary_crossentropy", optimizer=self.optimizer)
        gd.compile(loss="binary_crossentropy", optimizer=self.optimizer)
        d.trainable = True
        d.compile(loss="binary_crossentropy", optimizer=self.optimizer)

        self.generator = g
        self.discriminator = d
        self.gan = gd

    def train_model(self):
        """Train a created model

        Raises:
            Exception: Raised when no training data present.
            Exception: Raised when model is not present.
        """

        n_epochs, batch_size, dim_z = self._util.read_config(
            "epochs", "batch_size", "dim_z"
        )

        if self.data is None:
            raise Exception("No training data found, load data first")

        if self.generator is None or self.discriminator is None or self.gan is None:
            raise Exception("Model is not present. Create one first")

        X_train: np.array = self.data.X_train
        y_train: np.array = self.data.y_train

        self._util.log(
            "Training starting with:\nNumber of Epochs: {0}\nBatch Size: {1}\nNumber of Samples: {2}\n".format(
                n_epochs, batch_size, len(X_train)
            ),
            1,
        )

        g: "models.Model" = self.generator
        d: "models.Model" = self.discriminator
        gd: "models.Model" = self.gan

        iteration: int = 0

        nb_of_iterations_per_epoch: int = int(X_train.shape[0] / batch_size)
        self._util.log(
            "Number of iterations per epoch: {0}".format(nb_of_iterations_per_epoch), 1
        )

        disc_losses = list()
        gen_losses = list()

        for epoch in range(n_epochs):

            g_losses_for_epoch = []
            d_losses_for_epoch = []

            for i in range(nb_of_iterations_per_epoch):
                self._util.log(
                    "Epoch {0}/{1}, Iteration {2}/{3}".format(
                        epoch + 1, n_epochs, i + 1, nb_of_iterations_per_epoch
                    ),
                    1,
                )

                noise = util.generate_noise((batch_size, dim_z))

                image_batch = X_train[i * batch_size : (i + 1) * batch_size]
                label_batch = y_train[i * batch_size : (i + 1) * batch_size]

                generated_images = g.predict([noise, label_batch], verbose=0)

                # --Picture builder, for progress indication, not relevant atm--
                # if i % 5 == 0:
                #    image_grid = generate_mnist_image_grid(g, title="Epoch {0}, iteration {1}".format(epoch, iteration))
                #    save_generated_image(image_grid, epoch, i, "./images/generated_per_iteration")
                #    image_logger.log_images("generated_mnist_images_per_iteration", [image_grid], iteration)

                X = np.concatenate((image_batch, generated_images))
                y = [1] * batch_size + [0] * batch_size
                label_batches_for_discriminator = np.concatenate(
                    (label_batch, label_batch)
                )

                D_loss = d.train_on_batch([X, label_batches_for_discriminator], y)
                d_losses_for_epoch.append(D_loss)
                self._util.log("Discriminator Loss: {0}".format(D_loss), 1)

                noise = util.generate_noise((batch_size, dim_z))
                d.trainable = False
                G_loss = gd.train_on_batch([noise, label_batch], [1] * batch_size)
                d.trainable = True
                g_losses_for_epoch.append(G_loss)
                self._util.log("Generator Loss: {0}".format(G_loss), 1)

                iteration += 1

            # Save a generated image for every epoch
            classes = self._util.read_config("use_classes", strict=False)
            image_grid = util.generate_mnist_image_grid(
                g, dim_z, self._dim_y, classes=classes, title="Epoch {0}".format(epoch)
            )
            util.save_generated_image(
                image_grid, epoch, 0, "./images/generated_per_epoch"
            )

            self._util.log(
                "D loss: {0}, G loss: {1}".format(
                    np.mean(d_losses_for_epoch), np.mean(g_losses_for_epoch)
                ),
                1,
            )
            disc_losses.append(np.mean(d_losses_for_epoch, dtype="float"))
            gen_losses.append(np.mean(g_losses_for_epoch, dtype="float"))

        # Save losses
        train_history = dict()
        train_history["generator_loss"] = gen_losses
        train_history["discriminator_loss"] = disc_losses
        self._train_history = train_history

    def _generator_model(self) -> "models.Model":
        """Create GAN generator model

        Returns:
            "models.Model": The generator model.
        """
        dim_z = self._util.read_config("dim_z")

        # It is (at the moment) predefined that the images have to be squares
        sample_shape = self._data_shape
        fourth = sample_shape[0] // 4  # fourth of image width/height

        # New architecture (compare https://github.com/gaborvecsei/CDCGAN-Keras/blob/master/cdcgan/cdcgan_train.py)
        # ATTENTION: Concatenates z (noise) and y (condition) -> dim_y has to be the same as dim_z
        dim_y = self._dim_y

        # Prepare noise input
        input_z = layers.Input((dim_z,))
        dense_z_1 = layers.Dense(1024)(input_z)
        act_z_1 = layers.Activation("tanh")(dense_z_1)
        # act_z_1 = layers.Activation('relu')(dense_z_1)
        dense_z_2 = layers.Dense(128 * fourth * fourth)(act_z_1)
        bn_z_1 = layers.BatchNormalization()(dense_z_2)
        reshape_z = layers.Reshape(
            (fourth, fourth, 128), input_shape=(128 * fourth * fourth,)
        )(bn_z_1)

        # Prepare Conditional (label) input
        input_c = layers.Input((dim_y,))
        dense_c_1 = layers.Dense(1024)(input_c)
        act_c_1 = layers.Activation("tanh")(dense_c_1)
        # act_c_1 = layers.Activation('relu')(dense_c_1)
        dense_c_2 = layers.Dense(128 * fourth * fourth)(act_c_1)
        bn_c_1 = layers.BatchNormalization()(dense_c_2)
        reshape_c = layers.Reshape(
            (fourth, fourth, 128), input_shape=(128 * fourth * fourth,)
        )(bn_c_1)

        # Combine input source
        concat_z_c = layers.Concatenate()([reshape_z, reshape_c])

        # Image generation with the concatenated inputs
        up_1 = layers.UpSampling2D(size=(2, 2))(concat_z_c)
        # conv_1 = layers.Conv2D(64, (5, 5), padding='same')(up_1)
        conv_1 = layers.Conv2D(64, (2, 2), padding="same")(up_1)
        act_1 = layers.Activation("tanh")(conv_1)
        # act_1 = layers.Activation('relu')(conv_1)
        # TODO Use Conv2D_Transpose
        up_2 = layers.UpSampling2D(size=(2, 2))(act_1)
        conv_2 = layers.Conv2D(sample_shape[2], (5, 5), padding="same")(up_2)
        # conv_2 = layers.Conv2D(sample_shape[2], (2, 2), padding='same')(up_2)
        act_2 = layers.Activation("tanh")(conv_2)
        # act_2 = layers.Activation('sigmoid')(conv_2)
        model = models.Model(inputs=[input_z, input_c], outputs=act_2, name="Generator")
        return model

    def _discriminator_model(self) -> "models.Model":
        """Create GAN discriminator model

        Returns:
            "models.Model": The discriminator model.
        """

        # It is (at the moment) predefined that the images have to be squares
        sample_shape = self._data_shape
        fourth = sample_shape[0] // 4  # fourth of image width/height

        # New architecture (compare https://github.com/gaborvecsei/CDCGAN-Keras/blob/master/cdcgan/cdcgan_train.py)
        # ATTENTION: Concatenates z (noise) and y (condition) -> dim_y has to be the same as dim_z
        dim_y = self._dim_y

        input_gen_image = layers.Input(sample_shape)
        conv_1_image = layers.Conv2D(64, (5, 5), padding="same")(input_gen_image)
        act_1_image = layers.Activation("tanh")(conv_1_image)
        # act_1_image = layers.LeakyReLU(0.2)(conv_1_image)
        pool_1_image = layers.MaxPooling2D(pool_size=(2, 2))(act_1_image)
        conv_2_image = layers.Conv2D(128, (5, 5), padding="same")(pool_1_image)
        act_2_image = layers.Activation("tanh")(conv_2_image)
        # act_2_image = layers.LeakyReLU(0.2)(conv_2_image)
        pool_2_image = layers.MaxPooling2D(pool_size=(2, 2))(act_2_image)

        input_c = layers.Input((dim_y,))
        dense_1_c = layers.Dense(1024)(input_c)
        act_1_c = layers.Activation("tanh")(dense_1_c)
        # act_1_c = layers.LeakyReLU(0.2)(dense_1_c)
        dense_2_c = layers.Dense(fourth * fourth * 128)(act_1_c)
        bn_c = layers.BatchNormalization()(dense_2_c)
        reshaped_c = layers.Reshape((fourth, fourth, 128))(bn_c)

        concat = layers.Concatenate()([pool_2_image, reshaped_c])

        flat = layers.Flatten()(concat)
        dense_1 = layers.Dense(1024)(flat)
        act_1 = layers.Activation("tanh")(dense_1)
        # act_1 = layers.LeakyReLU(0.2)(dense_1)
        dense_2 = layers.Dense(1)(act_1)
        act_2 = layers.Activation("sigmoid")(dense_2)
        model = models.Model(
            inputs=[input_gen_image, input_c], outputs=act_2, name="Discriminator"
        )
        return model

    def _generator_containing_discriminator(
        self, g: "models.Model", d: "models.Model"
    ) -> "models.Model":
        """Combines generator and discriminator

        Args:
            g ("models.Model"): The generator model.
            d ("models.Model"): The discriminator model.

        Returns:
            "models.Model": The combined model.
        """
        dim_z = self._util.read_config("dim_z")

        input_z = layers.Input((dim_z,))

        # ATTENTION: Concatenates z (noise) and y (condition) -> dim_y has to be the same as dim_z
        input_y = layers.Input((self._dim_y,))
        gen_image = g([input_z, input_y])
        d.trainable = False
        is_real = d([gen_image, input_y])
        model = models.Model(inputs=[input_z, input_y], outputs=is_real, name="GAN")
        return model

    def save_model(self) -> None:
        """Save the weights of the trained GAN model

        Args:
            training. Strongly recommended. Defaults to True.

        """

        generator_entry = {"name": "generator", "model": self.generator}
        discriminator_entry = {"name": "discriminator", "model": self.discriminator}
        gan_entry = {"name": "gan", "model": self.gan}
        models = [generator_entry, discriminator_entry, gan_entry]

        super()._save_model(models)

    def load_model(self):
        """Load the weights of training GAN and create a model with them."""

        model_names = ["generator", "discriminator", "gan"]
        models = super()._load_model(model_names)

        self.generator = models[0]
        self.discriminator = models[1]
        self.gan = models[2]

        self.check_hash()


class VAEContainer(ModelContainer):
    """VAE base class"""

    # VAE specific variables
    encoder = None
    decoder = None
    vae = None

    def __init__(self, model_config: str, config: dict, verbose: int = 1):
        """VAE Constructor

        Do not call directly. Use ModelContainer.create to construct instances.

        Args:
            model_config (str): Name of the model config.
            verbose (int, optional): Verbosity level. Is used for logging. Defaults to 1.
        """
        self._type = "VAE"
        super().__init__(model_config, config, verbose)

    def get_early_stopping(self) -> bool:
        """Reads the flag for early stopping callback from config file and returns it.

        Returns:
            bool: Whether to use early stopping callback.
        """
        return self._util.read_config("early_stopping", strict=False)

    def get_latent_dim(self) -> int:
        """Reads the latent dim from config file and returns it.

        Returns:
            int: number of latent dimensions.
        """
        return self._util.read_config("dim_z", strict=True)

    def get_lstm_size(self) -> int:
        """Reads the lstm size from config file and returns it.

        Returns:
            int: Size of layer.
        """
        return self._util.read_config("lstm_size", strict=True)

    def get_lower_bound(self) -> float:
        """Reads lower bound from config file and returns it.

        Returns:
            float: Value of lower bound.
        """
        return self._util.read_config("ldp_lower_bound", strict=False)

    def get_log_gradients(self) -> bool:
        """Reads the flag for gradient logging from config file and returns it.

        Returns:
            bool: Whether to log gradients during training.
        """
        return self._util.read_config("log_gradients", strict=False)

    def get_latent_analysis(self) -> int or None:
        """Reads the epoch modulo from config file for LDA latent space analysis and returns it.

        Returns:
            int or None: Epoch modulo or None if callbacks should'nt be used.
        """
        return self._util.read_config("epoch_modulo", strict=False)

    def evaluate_gradient_file(
        self,
        save_figure: bool = True,
    ) -> None:

        log_gradients = self.get_log_gradients()

        gradient_file = os.path.join(self.get_model_dir(), "gradient_file")

        if not os.path.isfile(gradient_file):
            self._util.log(
                f"File {gradient_file} not found. Run training with `log_gradient` flag in config.",
                1,
            )
            return

        num_processes = len(os.sched_getaffinity(0))

        manager = Manager()
        res = manager.list()
        work = manager.Queue(num_processes)

        process_pool = []

        if log_gradients == "memory_efficient":

            def handle_line(in_queue, out_list):
                while True:
                    line = in_queue.get()
                    if line is None:
                        return
                    median, mean = line.split("\t")
                    out_list.append([float(median), float(mean)])

        elif log_gradients:

            def handle_line(in_queue, out_list):
                while True:
                    line = in_queue.get()
                    if line is None:
                        return
                    line = line.replace("[", "").replace("]", "").replace("\n", "")
                    arr = np.array([float(l) for l in line.split(" ")])
                    p25, p50, p75 = np.percentile(arr, [25, 50, 75])

                    arr = np.round(arr, 4)
                    val, num = np.unique(arr, return_counts=True)

                    out_list.append(
                        [
                            np.min(arr),
                            np.max(arr),
                            np.mean(arr),
                            p50,
                            p25,
                            p75,
                            val,
                            num,
                        ]
                    )

        else:
            self.log(
                "Called `evaluate_gradient_file` but log_gradients not set so nothing happened.",
                2,
            )
            return

        for _ in range(num_processes):
            p = Process(target=handle_line, args=(work, res))
            p.start()
            process_pool.append(p)

        with open(gradient_file, "r") as fr:
            it = itertools.chain(fr, (None,) * num_processes)
            for line in it:
                work.put(line)

        for p in process_pool:
            p.join()

        results = np.array(res)

        if log_gradients == "memory_efficient":
            self._config["median_clip"] = float(np.mean(results[:, 0]))
            self._config["mean_clip"] = float(np.mean(results[:, 1]))
            self._config["25_perc_clip"] = None
            self._config["75_perc_clip"] = None

        elif log_gradients:
            self._config["median_clip"] = float(np.mean(results[:, 3]))
            self._config["25_perc_clip"] = float(np.mean(results[:, 4]))
            self._config["75_perc_clip"] = float(np.mean(results[:, 5]))
            self._config["mean_clip"] = float(np.mean(results[:, 2]))

            if save_figure:
                savepath = os.path.join(
                    self.get_figure_dir(),
                    "gradient_distribution_over_training_steps.pdf",
                )

                figure_helper.plot_gradient_distribution(results, savepath)

                savepath = os.path.join(
                    self.get_figure_dir(), "gradient_distribution_over_count.pdf"
                )

                values, counts = np.concatenate(results[:, 6]), np.concatenate(
                    results[:, 7]
                )
                plot_results = {}

                for v, c in zip(values, counts):
                    if v not in plot_results:
                        plot_results[v] = 0
                    plot_results[v] += c

                figure_helper.plot_gradient_count(
                    list(plot_results.keys()), list(plot_results.values()), savepath
                )

        self.flush_config()

    def train_model(self, data: dict = None) -> None:
        """Trains the model.

        Args:
            data (dict, optional): Data to use for training. If None is passed use generators of model dataset. Defaults to None.

        Raises:
            RuntimeError: No data present.
            RuntimeError: No model present.
        """
        self._checkpoint_mode = False

        n_epochs, batch_size = self._util.read_config("epochs", "batch_size")

        if self.data is None and data is None:
            raise RuntimeError(
                f"{self.__class__} has no data and no data was passed. Load data or pass it in subclass."
            )

        if self.vae is None:
            raise RuntimeError(
                f"{self.__class__} has no model present. Create one first."
            )

        list_of_callbacks = []

        if self.get_early_stopping():
            list_of_callbacks.append(
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-10,
                    patience=10,
                    restore_best_weights=True,
                )
            )

        # TODO different handling of special callbacks, like data generation
        # i.e., introduce flag to determine which callbacks should be set

        # if data:
        #     # Custom callback for automated image printing
        #     print_image_callback = callbacks.LambdaCallback(
        #         on_epoch_end=lambda epoch, logs: util.generate_mnist_image_grid_vae(
        #             [self.encoder, self.decoder], [x_val, y_val], epoch, dim_z
        #         )
        #         if epoch % 1 == 0
        #         else None
        #     )
        #     list_of_callbacks.append(print_image_callback)

        if self.get_log_gradients:
            gradient_file = os.path.join(self.get_model_dir(), "gradient_file")
            if os.path.isfile(gradient_file):
                os.remove(gradient_file)

        if data:
            x_train = data["x_train"]
            x_val = data["x_val"]
            y_train = data["y_train"]
            y_val = data["y_val"]

            if chkpt_epochs := self.get_model_checkpoint():
                chkpt_dir = self.get_checkpoint_dir()

                # checkpoint callback checks after each step
                num_steps_per_epoch = int(np.ceil(len(x_train) / batch_size))

                list_of_callbacks.append(
                    callbacks.ModelCheckpoint(
                        filepath=os.path.join(
                            chkpt_dir, self._checkpoint_format_string
                        ),
                        save_best_only=False,
                        save_weights_only=True,
                        save_freq=chkpt_epochs * num_steps_per_epoch,
                    )
                )

            self._util.log(
                f"{self.__class__} starts training with:\n\tNumber of Epochs: {n_epochs}\n\tBatch Size: {batch_size}\n\t# train samples: {x_train.shape[0]}\n\t# validation samples: {x_val.shape[0]}\n",
                2,
            )

            if self._conditional:

                if epoch_modulo := self.get_latent_analysis():
                    list_of_callbacks.append(
                        model_helper.customLDACallback(
                            encoder=self.encoder,
                            data=[x_val, y_val],
                            labels=y_val,
                            batch_size=batch_size,
                            figure_dir=self.get_figure_dir(),
                            epoch_modulo=epoch_modulo,
                        )
                    )
                hist = self.vae.fit(
                    [x_train, y_train],
                    batch_size=batch_size,
                    epochs=n_epochs,
                    validation_data=([x_val, y_val], None),
                    callbacks=list_of_callbacks,
                )
            else:
                if epoch_modulo := self.get_latent_analysis():
                    list_of_callbacks.append(
                        model_helper.customPCACallback(
                            encoder=self.encoder,
                            data=[x_val, y_val],
                            labels=y_val,
                            batch_size=batch_size,
                            figure_dir=self.get_figure_dir(),
                            epoch_modulo=epoch_modulo,
                        )
                    )

                hist = self.vae.fit(
                    x_train,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    validation_data=(x_val, None),
                    callbacks=list_of_callbacks,
                )

        else:
            x_train, _, x_val = self.data.get_generators(batch_size, self._conditional)
            train_shape, _, val_shape = self.data.get_data_shape()

            self._util.log(
                f"{self.__class__} starts training with generators and:\n\tNumber of Epochs: {n_epochs}\n\tBatch Size: {batch_size}\n\t# train samples: {train_shape[0]}\n\t# validation samples: {val_shape[0]}\n",
                2,
            )

            if chkpt_epochs := self.get_model_checkpoint():
                chkpt_dir = self.get_checkpoint_dir()

                list_of_callbacks.append(
                    callbacks.ModelCheckpoint(
                        filepath=os.path.join(
                            chkpt_dir, self._checkpoint_format_string
                        ),
                        save_best_only=False,
                        save_weights_only=True,
                        save_freq=chkpt_epochs * len(x_train),
                    )
                )

            if epoch_modulo := self.get_latent_analysis():
                if self._conditional:
                    list_of_callbacks.append(
                        model_helper.customLDACallback(
                            encoder=self.encoder,
                            data=x_val,
                            labels=None,
                            batch_size=batch_size,
                            figure_dir=self.get_figure_dir(),
                            epoch_modulo=epoch_modulo,
                        )
                    )
                else:
                    list_of_callbacks.append(
                        model_helper.customPCACallback(
                            encoder=self.encoder,
                            data=x_val,
                            labels=None,
                            batch_size=batch_size,
                            figure_dir=self.get_figure_dir(),
                            epoch_modulo=epoch_modulo,
                        )
                    )
            hist = self.vae.fit(
                x_train,
                steps_per_epoch=len(x_train),
                epochs=n_epochs,
                validation_data=x_val,
                validation_steps=len(x_val),
                callbacks=list_of_callbacks,
            )
        self._train_history = hist.history
        self.plot_model_history()

        # self.evaluate_gradient_file()

        self.compute_epsilon_delta_with_rdp_accountant()
        # self.compute_epsilon_ldp()

    def save_model(self) -> None:
        """Save the weights of the trained VAE model

        Args:
            save_samples (bool, optional): Save data (partition) used for training. Strongly recommended. Defaults to True.

        """
        self._checkpoint_mode = False

        encoder_entry = {"name": "encoder", "model": self.encoder}
        decoder_entry = {"name": "decoder", "model": self.decoder}
        vae_entry = {"name": "vae", "model": self.vae}
        models = [encoder_entry, decoder_entry, vae_entry]

        super()._save_model(models)

    def load_model(self) -> None:
        """Load the weights of training GAN and create a model with them."""
        self._checkpoint_mode = False

        model_names = ["encoder", "decoder", "vae"]
        models = super()._load_model(model_names)

        self.encoder = models[0]
        self.decoder = models[1]
        self.vae = models[2]

        self.check_hash()

    def load_checkpoint_at(self, epoch: int) -> None:
        self.create_model(trained=True)
        self._checkpoint_mode = epoch

        self.vae.load_weights(
            os.path.join(
                self.get_checkpoint_dir(),
                self._checkpoint_format_string.format(epoch=epoch),
            ),
            by_name=True,
        )

        self.check_hash()

    def evaluate_loss(
        self,
        data: dict = None,
        train: bool = False,
        test: bool = True,
        val: bool = False,
        save: bool = True,
    ) -> dict:
        """Evaluates the model performance due to the loss on specified datasets.

        Args:
            data (dict, optional): Specific data to be used. If None, we fall back to data generators associated with the model. Defaults to None.
            train (bool, optional): Whether to evaluate on train data. Defaults to False.
            test (bool, optional): Whether to evaluate on test data. Defaults to True.
            val (bool, optional): Whether to evaluate on validation data. Defaults to False.
            save (bool, optional): Whether the result should be saved to `evaluation.json`. Defaults to True.
        """

        self._util.log(
            f"{self.__class__} starts evaluate_loss for: {'train ' if train else ''} {'test ' if test else ''} {'val ' if val else ''}",
            2,
        )

        # use evaluate function because different models may have different loss functions or weights
        # special handling for generators or numpy data

        batch_size = self._util.read_config("batch_size")

        if data:
            x_train = data["x_train"]
            x_test = data["x_test"]
            x_val = data["x_val"]
            y_train = data["y_train"]
            y_test = data["y_test"]
            y_val = data["y_val"]
        else:
            x_train, x_test, x_val = self.data.get_generators(
                batch_size, self._conditional
            )

        if data and self._conditional:
            train_loss = (
                self.vae.evaluate([x_train, y_train], batch_size=batch_size)
                if train
                else None
            )
            test_loss = (
                self.vae.evaluate([x_test, y_test], batch_size=batch_size)
                if test
                else None
            )
            val_loss = (
                self.vae.evaluate([x_val, y_val], batch_size=batch_size)
                if val
                else None
            )
        else:
            train_loss = (
                self.vae.evaluate(x_train, steps=len(x_train)) if train else None
            )
            test_loss = self.vae.evaluate(x_test, steps=len(x_test)) if test else None
            val_loss = self.vae.evaluate(x_val, steps=len(x_val)) if val else None

        result = {
            "train_loss": train_loss[0] if isinstance(train_loss, list) else train_loss,
            "test_loss": test_loss[0] if isinstance(test_loss, list) else test_loss,
            "val_loss": val_loss[0] if isinstance(val_loss, list) else val_loss,
        }

        if save:
            self.save_evaluation_metric(
                "losses",
                result,
            )

        return result

    def evaluate_image_reconstruction(
        self,
        data: dict = None,
        num_pictures: int = 25,
        train: bool = False,
        test: bool = True,
        val: bool = False,
    ) -> None:
        """Creates a image grid for the specified datasets where one can visually examine the reconstruction abilities of the vae.

        Args:
            data (dict, optional): Dictionary with data that should be used. Defaults to None.
            num_pictures (int, optional): If no data is specified, whe unravel the corresponding dataset und limit due to this. Defaults to 25.
            train (bool, optional): Whether to evaluate on train data. Defaults to False.
            test (bool, optional): Whether to evaluate on test data. Defaults to True.
            val (bool, optional): Whether to evaluate on validation data. Defaults to False.

        """

        if not isinstance(self.data, BaseImageClass):
            self._util.log(
                "evaluate_image_reconstruction expects image data. Skipped.", 2
            )
            return

        self._util.log(
            f"{self.__class__} starts evaluate_reconstruction for {num_pictures} figures and: {'train ' if train else ''} {'test ' if test else ''} {'val ' if val else ''}",
            2,
        )

        if data:
            x_train = data["x_train"]
            x_test = data["x_test"]
            x_val = data["x_val"]
            y_train = data["y_train"]
            y_test = data["y_test"]
            y_val = data["y_val"]
        else:
            x_train, x_test, x_val, y_train, y_test, y_val = self.data.unravel(
                limit=num_pictures, random_order=False
            )

        if self._conditional:
            pred_train = self.vae.predict([x_train, y_train]) if train else None
            pred_test = self.vae.predict([x_test, y_test]) if test else None
            pred_val = self.vae.predict([x_val, y_val]) if val else None
        else:
            pred_train = self.vae.predict(x_train) if train else None
            pred_test = self.vae.predict(x_test) if test else None
            pred_val = self.vae.predict(x_val) if val else None

        figure_path = self.get_figure_dir()

        if train:
            figure_helper.plot_pairwise_image_grid(
                x_train,
                pred_train,
                savepath=os.path.join(
                    figure_path, "train_pairwise_image_grid_{}x{}.pdf"
                ),
            )

        if test:
            figure_helper.plot_pairwise_image_grid(
                x_test,
                pred_test,
                savepath=os.path.join(
                    figure_path, "test_pairwise_image_grid_{}x{}.pdf"
                ),
            )

        if val:
            figure_helper.plot_pairwise_image_grid(
                x_val,
                pred_val,
                savepath=os.path.join(figure_path, "val_pairwise_image_grid_{}x{}.pdf"),
            )

    def evaluate_ssim(
        self,
        data: dict = None,
        train: bool = False,
        test: bool = True,
        val: bool = False,
        save: bool = True,
    ) -> dict:
        """Evaluates the model performance due to the ssim on specified datasets. Returns mean and std.

        Args:
            data (dict, optional): Specific data to be used. If None, we fall back to data generators associated with the model. Defaults to None.
            train (bool, optional): Whether to evaluate on train data. Defaults to False.
            test (bool, optional): Whether to evaluate on test data. Defaults to True.
            val (bool, optional): Whether to evaluate on validation data. Defaults to False.
        """

        if not isinstance(self.data, BaseImageClass):
            self._util.log("evaluate_ssim expects image data. Skipped.", 2)
            return

        self._util.log(
            f"{self.__class__} starts evaluate_ssim for: {'train ' if train else ''} {'test ' if test else ''} {'val ' if val else ''}",
            2,
        )

        if data:
            x_train = data["x_train"]
            x_test = data["x_test"]
            x_val = data["x_val"]
            y_train = data["y_train"]
            y_test = data["y_test"]
            y_val = data["y_val"]

            train_shape = x_train.shape if train else None
            test_shape = x_test.shape if test else None
            val_shape = x_test.shape if val else None

            if self._conditional:
                pred_train = self.vae.predict([x_train, y_train]) if train else None
                pred_test = self.vae.predict([x_test, y_test]) if test else None
                pred_val = self.vae.predict([x_val, y_val]) if val else None
            else:
                pred_train = self.vae.predict(x_train) if train else None
                pred_test = self.vae.predict(x_test) if test else None
                pred_val = self.vae.predict(x_val) if val else None

            if train:
                train_all_ssim = metric_helper.calc_ssim_for_batch(
                    x_train, pred_train, train_shape
                )
                train_ssim = [
                    np.mean(train_all_ssim, dtype=float),
                    np.std(train_all_ssim, dtype=float),
                ]

            else:
                train_ssim = [None, None]

            if test:
                test_all_ssim = metric_helper.calc_ssim_for_batch(
                    x_test, pred_test, test_shape
                )
                test_ssim = [
                    np.mean(test_all_ssim, dtype=float),
                    np.std(test_all_ssim, dtype=float),
                ]

            else:
                test_ssim = [None, None]

            if val:
                val_all_ssim = metric_helper.calc_ssim_for_batch(
                    x_val, pred_val, val_shape
                )
                val_ssim = [
                    np.mean(val_all_ssim, dtype=float),
                    np.std(val_all_ssim, dtype=float),
                ]

            else:
                val_ssim = [None, None]

        else:
            batch_size = self._util.read_config("batch_size", strict=True)

            x_train, x_test, x_val = self.data.get_generators(
                batch_size, self._conditional
            )
            train_shape, test_shape, val_shape = self.data.get_data_shape()

            train_all_ssim, test_all_ssim, val_all_ssim = [], [], []

            if train:
                for n in range(len(x_train)):
                    orig_train = x_train.__getitem__(n)
                    pred_train = self.vae.predict(orig_train)

                    train_all_ssim.append(
                        metric_helper.calc_ssim_for_batch(
                            orig_train[0], pred_train, pred_train.shape
                        )
                    )

                train_ssim = [
                    np.mean(np.concatenate(train_all_ssim), dtype=float),
                    np.std(np.concatenate(train_all_ssim), dtype=float),
                ]
            else:
                train_ssim = [None, None]

            if test:
                for n in range(len(x_test)):
                    orig_test = x_test.__getitem__(n)
                    pred_test = self.vae.predict(orig_test)

                    test_all_ssim.append(
                        metric_helper.calc_ssim_for_batch(
                            orig_test[0], pred_test, pred_test.shape
                        )
                    )

                test_ssim = [
                    np.mean(np.concatenate(test_all_ssim), dtype=float),
                    np.std(np.concatenate(test_all_ssim), dtype=float),
                ]
            else:
                test_ssim = [None, None]

            if val:
                for n in range(len(x_val)):
                    orig_val = x_test.__getitem__(n)
                    pred_val = self.vae.predict(orig_val)

                    val_all_ssim.append(
                        metric_helper.calc_ssim_for_batch(
                            orig_val[0], pred_val, pred_val.shape
                        )
                    )

                val_ssim = [
                    np.mean(np.concatenate(val_all_ssim), dtype=float),
                    np.std(np.concatenate(val_all_ssim), dtype=float),
                ]
            else:
                val_ssim = [None, None]

        result = {
            "train_ssim": train_ssim,
            "test_ssim": test_ssim,
            "val_ssim": val_ssim,
        }

        if save:
            self.save_evaluation_metric(
                "ssim",
                result,
            )

        return result

    def evaluate_generated_images(self, num_samples=225) -> None:

        if not isinstance(self.data, BaseImageClass):
            self._util.log("evaluate_generated_images expects image data. Skipped.", 2)
            return

        figure_path = self.get_figure_dir()

        labels, num_classes = None, None

        if self._conditional:
            num_classes = self.data.get_num_unique_labels()
            labels = np.array(
                np.random.default_rng().random(num_samples) * num_classes
            ).astype(int)

        gen_images = self.generate_new_data(num_samples, labels, num_classes)
        figure_helper.plot_image_grid(
            gen_images, os.path.join(figure_path, "generated_images_{}_x_{}.pdf")
        )

    def _test_if_keys_in_model_config(self, *args) -> bool:
        for key in args:
            if not key in self._config.keys():
                return False
        return True

    def _set_path_to_vgg_conf(self) -> str:

        if self._test_if_keys_in_model_config("epsilon", "train", "test", "val"):
            if (
                self._config["epsilon"] is not None
                and self._config["train"]
                and self._config["test"]
                and self._config["val"]
            ):
                self.path_to_vgg_conf = os.path.join(
                    self._base_path,
                    "/".join(ModelContainer._configs_dir.split("/")[:-1]),
                    "optimizer_configs",
                    "_".join(
                        [
                            "vgg16-" + self._config["dataset"],
                            "eps",
                            str(self._config["epsilon"]),
                            "m",
                            str(self._config["m"])
                            if self._test_if_keys_in_model_config("m")
                            else "64",
                            "b",
                            str(self._config["b"])
                            if self._test_if_keys_in_model_config("b")
                            else "1",
                        ]
                    )
                    + ".json",
                )
                return

        self.path_to_vgg_conf = os.path.join(
            self._base_path,
            "/".join(ModelContainer._configs_dir.split("/")[:-1]),
            "optimizer_configs",
            f"vgg16-{self._config['dataset']}.json",
        )

    def evaluate_against_vgg16(self, save: bool = True) -> dict:

        if not isinstance(self.data, BaseImageClass):
            self._util.log("evaluate_against_vgg16 expects image data. Skipped.", 2)
            return

        self._set_path_to_vgg_conf()

        if not os.path.isfile(self.path_to_vgg_conf):
            self._util.log(
                f"Need a config for vgg classifier, but {self.path_to_vgg_conf} not present. Skipped.",
                2,
            )
            return

        with open(self.path_to_vgg_conf, "r") as f:
            vgg_config = json.load(f)

            vgg_learning_rate, vgg_batch_size, vgg_epochs = (
                vgg_config["learning_rate"],
                vgg_config["batch_size"],
                vgg_config["epochs"],
            )

        num_classes = self.data.get_num_unique_labels()

        _, x_test, x_val, y_train, y_test, y_val = self.data.unravel()

        if y_train is None or y_test is None:
            self._util.log(
                "Evaluation with VGG only possible with conditional information. Skipped.",
                2,
            )
            return

        vgg_train_labels = np.argmax(y_train, axis=1)

        gen_train_data = self.generate_new_data(
            len(vgg_train_labels),
            labels=vgg_train_labels,
            num_classes=num_classes,
        )

        train_generator = SimpleConditionalDataGenerator(
            gen_train_data,
            y_train,
            vgg_batch_size,
            True,
        )

        test_generator = SimpleConditionalDataGenerator(
            x_test, y_test, vgg_batch_size, False
        )

        val_generator = SimpleConditionalDataGenerator(
            x_val, y_val, vgg_batch_size, True
        )

        vgg16 = model_helper.customVGG16Model(optimizer=Adam(vgg_learning_rate))
        vgg16.create(input_shape=self._data_shape, num_classes=num_classes)
        vgg16.fit(
            train_generator,
            epochs=vgg_epochs,
            batch_size=vgg_batch_size,
            val_generator=val_generator,
        )
        metrics = vgg16.full_evaluation(test_generator)

        figure_helper.plot_values_over_keys(
            data=vgg16._history.history,
            savepath=os.path.join(self.get_figure_dir(), "vgg16_training_history.pdf"),
        )

        figure_helper.plot_confusion_matrix(
            metrics["confusion_matrix"],
            np.arange(num_classes),
            savepath=os.path.join(self.get_figure_dir(), "vgg16_confusion_matrix.pdf"),
        )
        figure_helper.plot_confusion_matrix(
            metrics["confusion_matrix"],
            np.arange(num_classes),
            True,
            savepath=os.path.join(
                self.get_figure_dir(), "vgg16_confusion_matrix_normalized.pdf"
            ),
        )

        if save:
            self.save_evaluation_metric(
                "vgg16_classifier",
                metrics,
            )

        del vgg16
        K.clear_session()

        return metrics

    def _set_path_to_harcnn_conf(self):
        if self._test_if_keys_in_model_config("epsilon", "train", "test", "val"):
            if (
                self._config["epsilon"] is not None
                and self._config["train"]
                and self._config["test"]
                and self._config["val"]
            ):
                self.path_to_ms_conf = os.path.join(
                    self._base_path,
                    "/".join(ModelContainer._configs_dir.split("/")[:-1]),
                    "optimizer_configs",
                    f"harcnn-{self._config['dataset']}_eps_{self._config['epsilon']}.json",
                )
                return

        self.path_to_ms_conf = os.path.join(
            self._base_path,
            "/".join(ModelContainer._configs_dir.split("/")[:-1]),
            "optimizer_configs",
            f"harcnn-{self._config['dataset']}.json",
        )

    def evaluate_against_harcnn(self, save: bool = True) -> dict:

        if not isinstance(self.data, BaseTimeSeriesClass):
            self._util.log(
                "evaluate_against_harcnn expects time series data. Skipped.", 2
            )
            return

        self._set_path_to_harcnn_conf()

        if not os.path.isfile(self.path_to_ms_conf):
            self._util.log(
                f"Need a config for motion classifier, but {self.path_to_ms_conf} not present. Skipped.",
                2,
            )
            return

        with open(self.path_to_ms_conf, "r") as f:
            ms_config = json.load(f)

            ms_learning_rate, ms_batch_size, ms_epochs = (
                ms_config["learning_rate"],
                ms_config["batch_size"],
                ms_config["epochs"],
            )

        num_classes = self.data.get_num_unique_labels()

        _, x_test, x_val, y_train, y_test, y_val = self.data.unravel()

        if y_train is None or y_test is None:
            self._util.log(
                "Evaluation of time series only possible with conditional information. Skipped.",
                2,
            )
            return

        harcnn_train_labels = np.argmax(y_train, axis=1)

        gen_train_data = self.generate_new_data(
            len(harcnn_train_labels),
            labels=harcnn_train_labels,
            num_classes=num_classes,
        )

        train_generator = SimpleConditionalDataGenerator(
            gen_train_data,
            y_train,
            ms_batch_size,
            True,
        )

        test_generator = SimpleConditionalDataGenerator(
            x_test, y_test, ms_batch_size, False
        )

        val_generator = SimpleConditionalDataGenerator(
            x_val, y_val, ms_batch_size, True
        )

        motion_classifier = model_helper.customHARCNNModel(
            optimizer=Adam(ms_learning_rate)
        )
        motion_classifier.create(input_shape=self._data_shape, num_classes=num_classes)
        motion_classifier.fit(
            train_generator,
            epochs=ms_epochs,
            batch_size=ms_batch_size,
            val_generator=val_generator,
        )
        metrics = motion_classifier.full_evaluation(test_generator)

        figure_helper.plot_values_over_keys(
            data=motion_classifier._history.history,
            savepath=os.path.join(self.get_figure_dir(), "training_history_harcnn.pdf"),
        )

        figure_helper.plot_confusion_matrix(
            metrics["confusion_matrix"],
            np.arange(num_classes),
            savepath=os.path.join(self.get_figure_dir(), "harcnn_confusion_matrix.pdf"),
        )
        figure_helper.plot_confusion_matrix(
            metrics["confusion_matrix"],
            np.arange(num_classes),
            True,
            savepath=os.path.join(
                self.get_figure_dir(), "harcnn_confusion_matrix_normalized.pdf"
            ),
        )

        if save:
            self.save_evaluation_metric(
                "harcnn_classifier",
                metrics,
            )

        del motion_classifier
        K.clear_session()

        return metrics

    def generate_new_data(
        self, sample_size: int = 1, labels: list or int = None, num_classes: int = None
    ) -> list:
        """This function generates new data.

        Args:
            sample_size (int, optional): Number of random new samples to generate. Defaults to 1.
            labels (list or int, optional): For conditional models the target labels. Defaults to None.
            num_classes (int, optional): For conditional models the number of overall labels. If labels are set but None this is inferred from passed labels (max). Defaults to None.

        Raises:
            RuntimeError: Model not present.
            RuntimeError: Missing label information.

        Returns:
            list: Newly generated data.
        """

        if self.vae is None:
            raise RuntimeError(
                f"{self.__class__} has no model present. Create one first to generate data."
            )

        # dim_z = self.decoder.input_shape[-1]
        # mean = np.zeros(dim_z)
        # cov = np.eye(dim_z)
        # z = np.random.default_rng().multivariate_normal(mean, cov, sample_size)

        # if self._conditional and (dim_z > self.get_latent_dim()):
        #     if labels is None:
        #         raise RuntimeError(
        #             f"To generate data for {self.__class__} pass label information."
        #         )
        #     labels = util.check_if_list_and_matches_length(
        #         labels, sample_size, "labels"
        #     )
        #     if num_classes is None:
        #         num_classes = np.max(labels)

        #     dim_z = dim_z - num_classes
        #     z = np.random.default_rng().normal(size=(sample_size, dim_z))
        #     labels = utils.to_categorical(labels, num_classes=num_classes)
        #     z = np.hstack((z, labels))

        # else:
        #     z = np.random.default_rng().normal(size=(sample_size, dim_z))

        # return self.decoder(z)

        _, _, x, _, _, y = self.data.unravel()
        y_labels = np.argmax(y, axis=1)

        if self._conditional:
            if labels is None:
                raise RuntimeError(
                    f"To generate data for {self.__class__} pass label information."
                )

            labels = util.check_if_list_and_matches_length(
                labels, sample_size, "labels"
            )

            global_data = np.zeros((sample_size, *x.shape[1:]))
            list_of_labels, counts = np.unique(labels, return_counts=True)

            for label, count in zip(list_of_labels, counts):
                label_data = []
                data_mask = y_labels == label
                train_generator = SimpleDataGenerator(
                    x[data_mask], y[data_mask], self._config["batch_size"]
                )
                while True:
                    tmp_pred = self.vae.predict(
                        train_generator, steps=len(train_generator)
                    )
                    label_data += list(tmp_pred)

                    if len(label_data) >= count:
                        break

                label_data = np.array(label_data)[:count]
                global_data[labels == label] = label_data

        else:
            global_data = []
            train_generator = SimpleDataGenerator(x, None, self._config["batch_size"])
            while True:
                tmp_pred = self.vae.predict(train_generator, steps=len(train_generator))
                global_data += list(tmp_pred)

                if len(global_data) >= sample_size:
                    break
            global_data = np.array(global_data)[:sample_size]

        return global_data

    def perturb_own_dataset(self) -> None:

        lower_bound = self.get_lower_bound()

        if lower_bound is None or lower_bound == 0.0:
            return

        if self.data is None:
            raise RuntimeError(
                f"{self.__class__} has no data present. Load data first."
            )

        if self.vae is None:
            raise RuntimeError(
                f"{self.__class__} has no model present. Create and load model first."
            )

        data_dir, dataset, batch_size = self._util.read_config(
            "data_dir", "dataset", "batch_size"
        )
        orig_data_path = os.path.join(data_dir, dataset, f"{dataset}.npz")

        X, y = DataContainer.load_data(orig_data_path)

        # TODO refactor dataset to allow better preprocessing
        y_cat = utils.to_categorical(y) if y is not None else None

        if isinstance(self.data, BaseImageClass):
            X = util.transform_images_zero_one_range(X)
        if isinstance(self.data, BaseTimeSeriesClass):
            if not (X.shape[1] == 12) and not (X.shape[2] == 500):
                X = X.reshape(-1, 12, 500)

        data_generator = SimpleDataGenerator(X, y_cat, batch_size, self._conditional)

        X_perturbed = self.vae.predict(data_generator, steps=len(data_generator))

        if isinstance(self.data, BaseImageClass):
            X_perturbed = util.inverse_transform_images_zero_one_range(X_perturbed)

        data_path = os.path.join(
            data_dir, dataset, f"{dataset}_noise_{lower_bound}.npz"
        )

        DataContainer.save_data(data_path, X_perturbed, y)

    def compute_epsilon_ldp(self, num_samples: int) -> Tuple[float, float]:

        dim_z, lower_bound = self.get_latent_dim(), self.get_lower_bound()

        return model_helper.compute_epsilon_delta_for_ldp(
            lower_bound, dim_z, num_samples
        )

    @abstractmethod
    def create_model(self) -> None:

        if not self.vae:
            raise ValueError(
                f"{self.__class__} misses vae. You have to build and set it in the subclass!"
            )

        if self.encoder:
            self.encoder.summary(print_fn=lambda x: self._util.log(x, 2))

        if self.decoder:
            self.decoder.summary(print_fn=lambda x: self._util.log(x, 2))

        self.set_optimizer()

        self.vae.compile(optimizer=self.optimizer)
        self.vae.summary(print_fn=lambda x: self._util.log(x, 2))


class Cifar10VAE(VAEContainer):
    def __init__(self, model_config: str, config: dict, verbose: int = 1):

        self._conditional = False
        super().__init__(model_config, config, verbose)

    def create_model(self, trained: bool = False) -> None:
        """Create VAE model

        Args:
            trained (bool, optional): Indicates if model was already trained before.
                    Not used in GAN subclass. Defaults to False.

        Raises:
            Exception: Raised when data is not present and data shape cannot be determined.
        """

        dim_z = self.get_latent_dim()

        # Alternative architecture for cifar10 VAEs
        # Directly copied from here:
        # https://github.wdf.sap.corp/D043326/membership_inf_gan_vae/blob/master/Monte-Carlo-Attacks/Monte-Carlo-CIFAR_VAE/cifar10_train.py

        original_img_size = (32, 32, 3)
        img_rows, img_cols, img_chns = original_img_size
        filters = 32
        kernel_size = 3
        intermediate_dim = 300
        latent_dim = dim_z

        x = layers.Input(shape=original_img_size)
        enc_conv_1 = layers.Conv2D(
            img_chns, kernel_size=(2, 2), padding="same", activation="relu"
        )(x)
        enc_conv_2 = layers.Conv2D(
            filters,
            kernel_size=(2, 2),
            padding="same",
            activation="relu",
            strides=(2, 2),
        )(enc_conv_1)
        enc_conv_3 = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            strides=1,
        )(enc_conv_2)
        enc_conv_4 = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            strides=1,
        )(enc_conv_3)
        enc_flat_layer = layers.Flatten()(enc_conv_4)
        enc_hidden_layer = layers.Dense(intermediate_dim, activation="relu")(
            enc_flat_layer
        )

        # mean and variance for latent variables
        z_mean = layers.Dense(latent_dim)(enc_hidden_layer)
        z_log_var = layers.Dense(latent_dim)(enc_hidden_layer)

        z = layers.Lambda(model_helper.sampling, output_shape=(latent_dim,))(
            [z_mean, z_log_var]
        )

        # decoder architecture
        decoder_hid = layers.Dense(int(intermediate_dim), activation="relu")
        decoder_upsample = layers.Dense(
            int(filters * img_rows / 2 * img_cols / 2), activation="relu"
        )

        if K.image_data_format() == "channels_first":
            output_shape = (filters, int(img_rows / 2), int(img_cols / 2))
        else:
            output_shape = (int(img_rows / 2), int(img_cols / 2), filters)

        decoder_reshape = layers.Reshape(output_shape)
        decoder_deconv_1 = layers.Conv2DTranspose(
            filters,
            kernel_size=kernel_size,
            padding="same",
            strides=1,
            activation="relu",
        )
        decoder_deconv_2 = layers.Conv2DTranspose(
            filters,
            kernel_size=kernel_size,
            padding="same",
            strides=1,
            activation="relu",
        )
        decoder_deconv_3_upsamp = layers.Conv2DTranspose(
            filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="valid",
            activation="relu",
        )
        decoder_mean_squash = layers.Conv2D(
            img_chns, kernel_size=2, padding="valid", activation="sigmoid"
        )

        hid_decoded = decoder_hid(z)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

        # Custom loss layer
        class CustomVariationalLayer(layers.Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(CustomVariationalLayer, self).__init__(**kwargs)

            def vae_loss(self, x, x_decoded_mean_squash):
                x = K.flatten(x)
                x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
                xent_loss = (
                    img_rows
                    * img_cols
                    * metrics.binary_crossentropy(x, x_decoded_mean_squash)
                )
                kl_loss = -0.5 * K.mean(
                    1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
                )
                return K.mean(xent_loss + kl_loss)

            def call(self, inputs):
                x = inputs[0]
                x_decoded_mean_squash = inputs[1]
                loss = self.vae_loss(x, x_decoded_mean_squash)
                self.add_loss(loss, inputs=inputs)
                return x

        y = layers.CustomVariationalLayer()([x, x_decoded_mean_squash])

        # Encoder and decoder will only be constructed here when
        # model is reconstructed before attacking (after the model has already been trained)
        # When the VAE is first constructed the encoder and decoder models will be built in the train method after successful training
        if trained:
            encoder = models.Model(x, z_mean)

            decoder_input = layers.Input(shape=(latent_dim,))
            _hid_decoded = decoder_hid(decoder_input)
            _up_decoded = decoder_upsample(_hid_decoded)
            _reshape_decoded = decoder_reshape(_up_decoded)
            _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
            _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
            _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
            _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
            decoder = models.Model(decoder_input, _x_decoded_mean_squash)
            self.encoder = encoder
            self.decoder = decoder
        else:
            # Save layers, so that encoder and decoder can be built later (after training)
            layers_dict = dict()
            layers_dict["x"] = x
            layers_dict["z_mean"] = z_mean
            layers_dict["decoder_hid"] = decoder_hid
            layers_dict["decoder_upsample"] = decoder_upsample
            layers_dict["decoder_reshape"] = decoder_reshape
            layers_dict["decoder_deconv_1"] = decoder_deconv_1
            layers_dict["decoder_deconv_2"] = decoder_deconv_2
            layers_dict["decoder_deconv_3_upsamp"] = decoder_deconv_3_upsamp
            layers_dict["decoder_mean_squash"] = decoder_mean_squash
            self._layers_dict = layers_dict

            self.encoder = None
            self.decoder = None

        # entire model
        self.vae = models.Model(x, y)
        super().create_model()

    def train_model(self) -> None:

        super().train_model()

        # Build encoder and decoder from saved layers
        layers_dict = self._layers_dict

        encoder = models.Model(layers["x"], layers["z_mean"])

        latent_dim = self.get_latent_dim()

        decoder_input = layers.Input(shape=(latent_dim,))
        _hid_decoded = layers_dict["decoder_hid"](decoder_input)
        _up_decoded = layers_dict["decoder_upsample"](_hid_decoded)
        _reshape_decoded = layers_dict["decoder_reshape"](_up_decoded)
        _deconv_1_decoded = layers_dict["decoder_deconv_1"](_reshape_decoded)
        _deconv_2_decoded = layers_dict["decoder_deconv_2"](_deconv_1_decoded)
        _x_decoded_relu = layers_dict["decoder_deconv_3_upsamp"](_deconv_2_decoded)
        _x_decoded_mean_squash = layers_dict["decoder_mean_squash"](_x_decoded_relu)
        decoder = models.Model(decoder_input, _x_decoded_mean_squash)

        self.encoder = encoder
        self.decoder = decoder


class DFCPictureVAE(VAEContainer):
    def __init__(self, model_config: str, config: dict, verbose: int = 1):

        self._conditional = False
        super().__init__(model_config, config, verbose)

    def create_model(self, trained: bool = False) -> None:

        self.check_if_data_shape_is_present()
        latent_size = self.get_latent_dim()
        input_shape = self._data_shape

        # loss weighting parameters
        alpha = 0.5  # kl-loss
        beta = 1  # reconstruction-loss

        # encoder params
        encoder_filters = [32, 64, 128, 256]
        encoder_kernels = (4, 4)
        encoder_stride = 2
        encoder_batch_norm = True
        encoder_num_layers = 4

        # decoder params
        decoder_num_layers = 4
        decoder_filters = [128, 64, 32, 3]
        decoder_kernels = (3, 3)
        decoder_stride = 1
        decoder_batch_norm = [True, True, True, False]
        decoder_activations = [True, True, True, False]
        decoder_upsampling_size = (2, 2)

        # build encoder
        encoder_inp = layers.Input(shape=input_shape, name="encoder_Input")
        encoder_deep = model_helper.get_deep_conv(
            "encoder",
            encoder_num_layers,
            encoder_inp,
            encoder_filters,
            encoder_kernels,
            encoder_stride,
            encoder_batch_norm,
        )
        encoder_flat = layers.Flatten(name="encoder_Flatten")(encoder_deep)
        latent_mu = layers.Dense(latent_size, name="latent_mu")(encoder_flat)
        latent_sigma = layers.Dense(latent_size, name="latent_sigma")(encoder_flat)
        z = layers.Lambda(model_helper.sampling, output_shape=(latent_size,), name="z")(
            [latent_mu, latent_sigma]
        )

        encoder = models.Model(
            encoder_inp, [latent_mu, latent_sigma, z], name="encoder"
        )

        # build decoder
        decoder_inp = layers.Input(z.shape[1:], name="decoder_Input")
        decoder_dense = layers.Dense(4096, name="decoder_Dense")(decoder_inp)
        decoder_reshape = layers.Reshape((4, 4, 256), name="decoder_Reshape")(
            decoder_dense
        )

        decoder_outp = model_helper.get_deep_conv_with_upsampling(
            "decoder",
            decoder_num_layers,
            decoder_reshape,
            decoder_upsampling_size,
            decoder_filters,
            decoder_kernels,
            decoder_stride,
            decoder_batch_norm,
            decoder_activations,
        )

        decoder = models.Model(decoder_inp, decoder_outp, name="decoder")
        vae_outp = decoder(encoder(encoder_inp)[-1])
        vae = models.Model(encoder_inp, vae_outp, name="vae")

        selected_vgg_layers = ["block1_conv1", "block1_conv2", "block2_conv1"]

        vgg19 = tf.keras.applications.VGG19(include_top=True, weights="imagenet")

        """Perceptual loss for the DFC VAE"""
        outputs = [vgg19.get_layer(l).output for l in selected_vgg_layers]
        model = models.Model(vgg19.input, outputs)

        encoder_inp = tf.image.resize(encoder_inp, [224, 224])
        vae_outp = tf.image.resize(vae_outp, [224, 224])

        h1_list = model(encoder_inp)
        h2_list = model(vae_outp)

        reconstruction_loss = 0.0

        for h1, h2 in zip(h1_list, h2_list):
            h1 = K.batch_flatten(h1)
            h2 = K.batch_flatten(h2)
            reconstruction_loss = reconstruction_loss + K.sum(
                K.square(h1 - h2), axis=-1
            )

        kl_loss = -0.5 * K.sum(
            1 + latent_sigma - K.square(latent_mu) - K.exp(latent_sigma),
            axis=-1,
        )

        vae_loss = beta * reconstruction_loss + alpha * kl_loss
        vae.add_loss(vae_loss if self.get_per_example_loss() else K.mean(vae_loss))

        self.encoder = encoder
        self.decoder = decoder
        self.vae = vae

        super().create_model()

    def train_model(self) -> None:
        super().train_model()


class BasicPictureVAE(VAEContainer):
    def __init__(self, model_config: str, config: dict, verbose: int = 1):

        self._conditional = True
        super().__init__(model_config, config, verbose)

    def create_model(self, trained: bool = False) -> None:
        """Create VAE model

        Args:
            trained (bool, optional): Indicates if model was already trained before.
                    Not used in GAN subclass. Defaults to False.

        Raises:
            Exception: Raised when data is not present and data shape cannot be determined.
        """
        self.check_if_data_shape_is_present()
        dim_y = self.data.get_num_unique_labels()
        dim_z = self.get_latent_dim()

        # Taken from keras VAE Example https://keras.io/examples/generative/vae/
        # Altered to conditional VAE
        flat_img_size = np.prod(self._data_shape)  # [:2])
        intermediate_dim = 512

        # Seperate inputs for sample and label
        input_x = layers.Input(shape=(flat_img_size,), name="Input_x")
        input_y = layers.Input(shape=(dim_y,), name="Input_y")

        # VAE model = encoder + decoder
        # build encoder model

        # Concatenate layer instead of single input layer
        # inputs = layers.Input(shape=input_shape, name='encoder_input')
        inputs = layers.concatenate([input_x, input_y], axis=1)

        x = layers.Dense(intermediate_dim, activation="relu")(inputs)
        z_mean = layers.Dense(dim_z, name="z_mean")(x)
        z_log_var = layers.Dense(dim_z, name="z_log_var")(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = layers.Lambda(model_helper.sampling, output_shape=(dim_z,), name="z")(
            [z_mean, z_log_var]
        )
        z_cond = layers.concatenate([z, input_y], axis=1)

        # instantiate encoder model
        encoder = models.Model(
            [input_x, input_y], [z_mean, z_log_var, z_cond], name="encoder"
        )

        # build decoder model
        latent_inputs = layers.Input(shape=(dim_z + dim_y,), name="z_sampling")
        x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
        outputs = layers.Dense(flat_img_size, activation="sigmoid")(x)

        # instantiate decoder model
        decoder = models.Model(latent_inputs, outputs, name="decoder")

        # instantiate VAE model
        outputs = decoder(encoder([input_x, input_y])[2])

        if log_gradients := self.get_log_gradients():
            gradient_file = os.path.join(self.get_model_dir(), "gradient_file")

            if log_gradients == "memory_efficient":
                cvae = model_helper.MemoryEfficientGradientModel(
                    [input_x, input_y],
                    outputs,
                    name="vae_mlp",
                    gradient_file=gradient_file,
                )
            else:
                cvae = model_helper.GradientModel(
                    [input_x, input_y],
                    outputs,
                    name="vae_mlp",
                    gradient_file=gradient_file,
                )
        else:
            cvae = models.Model([input_x, input_y], outputs, name="vae_mlp")

        reconstruction_loss = (
            losses.binary_crossentropy(input_x, outputs) * flat_img_size
        )
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)

        vae_loss = reconstruction_loss + kl_loss
        cvae.add_loss(vae_loss if self.get_per_example_loss() else K.mean(vae_loss))

        self.encoder = encoder
        self.decoder = decoder
        self.vae = cvae

        super().create_model()

    def train_model(self) -> None:

        x_train, _, x_val, y_train, _, y_val = self.data.unravel()

        # ? reshape
        # TODO How to handle color channel (cifar)?
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_val = x_val.reshape((len(x_val), np.prod(x_val.shape[1:])))

        data = {"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val}

        super().train_model(data)


class PlainPictureVAE(VAEContainer):
    def __init__(self, model_config: str, config: dict, verbose: int = 1):
        self._conditional = False
        super().__init__(model_config, config, verbose)

    def create_model(self, trained: bool = False) -> None:

        self.check_if_data_shape_is_present()
        latent_size = self.get_latent_dim()
        input_shape = self._data_shape

        # loss weighting parameters
        alpha = 1  # kl-loss
        beta = 1  # reconstruction-loss

        # encoder params
        encoder_filters = [32, 64, 128, 256]
        encoder_kernels = (4, 4)
        encoder_stride = 2
        encoder_batch_norm = False  # True
        encoder_num_layers = 4

        # decoder params
        decoder_num_layers = 4
        decoder_filters = [128, 64, 32, 3]
        decoder_kernels = (3, 3)
        decoder_stride = 1
        decoder_batch_norm = False  # [True, True, True, False]
        decoder_activations = [True, True, True, False]
        decoder_upsampling_size = (2, 2)

        # build encoder
        encoder_inp = layers.Input(shape=input_shape, name="encoder_Input")
        encoder_deep = model_helper.get_deep_conv(
            "encoder",
            encoder_num_layers,
            encoder_inp,
            encoder_filters,
            encoder_kernels,
            encoder_stride,
            encoder_batch_norm,
        )
        encoder_flat = layers.Flatten(name="encoder_Flatten")(encoder_deep)
        latent_mu = layers.Dense(latent_size, name="latent_mu")(encoder_flat)
        latent_sigma = layers.Dense(latent_size, name="latent_sigma")(encoder_flat)
        z = layers.Lambda(model_helper.sampling, output_shape=(latent_size,), name="z")(
            [latent_mu, latent_sigma]
        )

        encoder = models.Model(
            encoder_inp, [latent_mu, latent_sigma, z], name="encoder"
        )

        # build decoder
        decoder_inp = layers.Input(z.shape[1:], name="decoder_Input")
        decoder_dense = layers.Dense(4096, name="decoder_Dense")(decoder_inp)
        decoder_reshape = layers.Reshape((4, 4, 256), name="decoder_Reshape")(
            decoder_dense
        )

        decoder_outp = model_helper.get_deep_conv_with_upsampling(
            "decoder",
            decoder_num_layers,
            decoder_reshape,
            decoder_upsampling_size,
            decoder_filters,
            decoder_kernels,
            decoder_stride,
            decoder_batch_norm,
            decoder_activations,
        )

        decoder = models.Model(decoder_inp, decoder_outp, name="decoder")

        vae_outp = decoder(encoder(encoder_inp)[-1])

        if log_gradients := self.get_log_gradients():
            gradient_file = os.path.join(self.get_model_dir(), "gradient_file")

            if log_gradients == "memory_efficient":
                vae = model_helper.MemoryEfficientGradientModel(
                    encoder_inp,
                    vae_outp,
                    name="vae",
                    gradient_file=gradient_file,
                )
            else:
                vae = model_helper.GradientModel(
                    encoder_inp,
                    vae_outp,
                    name="vae",
                    gradient_file=gradient_file,
                )
        else:
            vae = models.Model(encoder_inp, vae_outp, name="vae")

        flat_input = tf.reshape(encoder_inp, (-1, np.prod(input_shape)))
        flat_prediction = tf.reshape(vae_outp, (-1, np.prod(input_shape)))
        # reconstruction_loss = losses.mean_squared_logarithmic_error(
        #     flat_input, flat_prediction
        # ) * np.prod(input_shape)

        reconstruction_loss = losses.mean_squared_error(
            flat_input, flat_prediction
        ) * np.prod(input_shape)

        kl_loss = -0.5 * K.sum(
            1 + latent_sigma - K.square(latent_mu) - K.exp(latent_sigma),
            axis=-1,
        )

        vae_loss = beta * reconstruction_loss + alpha * kl_loss
        vae.add_loss(vae_loss if self.get_per_example_loss() else K.mean(vae_loss))

        self.encoder = encoder
        self.decoder = decoder
        self.vae = vae

        super().create_model()

    def train_model(self) -> None:
        super().train_model()


class ConditionalPlainPictureVAE(VAEContainer):
    def __init__(self, model_config: str, config: dict, verbose: int = 1):
        self._conditional = True
        super().__init__(model_config, config, verbose)

    def create_model(self, trained: bool = False) -> None:

        self.check_if_data_shape_is_present()
        latent_size = self.get_latent_dim()
        input_shape = self._data_shape
        dim_y = self.data.get_num_unique_labels()
        lower_bound = self.get_lower_bound()

        # loss weighting parameters
        alpha = 1  # kl-loss
        beta = 1  # reconstruction-loss

        # encoder params
        encoder_filters = [32, 64, 128, 256]
        encoder_kernels = (4, 4)
        encoder_stride = 2
        encoder_batch_norm = False
        encoder_num_layers = 4

        # decoder params
        decoder_num_layers = 4
        decoder_filters = [128, 64, 32, 3]
        decoder_kernels = (3, 3)
        decoder_stride = 1
        decoder_batch_norm = False  # [True, True, True, False]
        decoder_activations = [True, True, True, False]
        decoder_upsampling_size = (2, 2)

        # build encoder
        encoder_inp = layers.Input(shape=input_shape, name="encoder_input")
        label_inp = layers.Input(shape=(dim_y,), name="label_input")
        encoder_deep = model_helper.get_deep_conv(
            "encoder",
            encoder_num_layers,
            encoder_inp,
            encoder_filters,
            encoder_kernels,
            encoder_stride,
            encoder_batch_norm,
        )
        encoder_flat = layers.Flatten(name="encoder_Flatten")(encoder_deep)
        # conc_enc_flat = layers.concatenate([encoder_flat, label_inp], axis=1)
        conc_enc_flat = encoder_flat

        if (lower_bound is not None) and (lower_bound > 0):
            latent_mu = layers.Dense(
                latent_size,
                name="latent_mu",
                activation=model_helper.custom_scaled_tanh,
            )(conc_enc_flat)
            latent_sigma = layers.Dense(
                latent_size,
                name="latent_sigma",
                activation=model_helper.sigma_bound(lower_bound=lower_bound),
            )(conc_enc_flat)
        else:
            latent_mu = layers.Dense(latent_size, name="latent_mu")(conc_enc_flat)
            latent_sigma = layers.Dense(latent_size, name="latent_sigma")(conc_enc_flat)

        z = layers.Lambda(model_helper.sampling, output_shape=(latent_size,), name="z")(
            [latent_mu, latent_sigma]
        )
        z = layers.concatenate([z, label_inp], axis=1)

        encoder = models.Model(
            [encoder_inp, label_inp], [latent_mu, latent_sigma, z], name="encoder"
        )

        # build decoder
        decoder_inp = layers.Input(z.shape[1:], name="decoder_input")
        decoder_dense = layers.Dense(4096, name="decoder_Dense")(decoder_inp)
        decoder_reshape = layers.Reshape((4, 4, 256), name="decoder_Reshape")(
            decoder_dense
        )

        decoder_outp = model_helper.get_deep_conv_with_upsampling(
            "decoder",
            decoder_num_layers,
            decoder_reshape,
            decoder_upsampling_size,
            decoder_filters,
            decoder_kernels,
            decoder_stride,
            decoder_batch_norm,
            decoder_activations,
        )

        decoder = models.Model(decoder_inp, decoder_outp, name="decoder")

        vae_outp = decoder(encoder([encoder_inp, label_inp])[-1])

        if log_gradients := self.get_log_gradients():
            gradient_file = os.path.join(self.get_model_dir(), "gradient_file")

            if log_gradients == "memory_efficient":
                vae = model_helper.MemoryEfficientGradientModel(
                    [encoder_inp, label_inp],
                    vae_outp,
                    name="vae",
                    gradient_file=gradient_file,
                )
            else:
                vae = model_helper.GradientModel(
                    [encoder_inp, label_inp],
                    vae_outp,
                    name="vae",
                    gradient_file=gradient_file,
                )
        else:
            vae = models.Model([encoder_inp, label_inp], vae_outp, name="vae")

        flat_input = tf.reshape(encoder_inp, (-1, np.prod(input_shape)))
        flat_prediction = tf.reshape(vae_outp, (-1, np.prod(input_shape)))

        reconstruction_loss = losses.mean_squared_error(
            flat_input, flat_prediction
        ) * np.prod(input_shape)

        kl_loss = -0.5 * K.sum(
            1 + latent_sigma - K.square(latent_mu) - K.exp(latent_sigma),
            axis=-1,
        )

        vae_loss = beta * reconstruction_loss + alpha * kl_loss
        vae.add_loss(vae_loss if self.get_per_example_loss() else K.mean(vae_loss))

        self.encoder = encoder
        self.decoder = decoder
        self.vae = vae

        super().create_model()

    def train_model(self) -> None:
        super().train_model()


class MultitaskPlainPictureVAE(VAEContainer):
    def __init__(self, model_config: str, config: dict, verbose: int = 1):
        self._conditional = True
        super().__init__(model_config, config, verbose)

    def create_model(self, trained: bool = False) -> None:

        self.check_if_data_shape_is_present()
        latent_size = self.get_latent_dim()
        input_shape = self._data_shape
        dim_y = self.data.get_num_unique_labels()

        # loss weighting parameters
        alpha = 1  # kl-loss
        beta = 1  # reconstruction-loss
        gamma = 1  # classifier

        # encoder params
        encoder_filters = [32, 64, 128, 256]
        encoder_kernels = (4, 4)
        encoder_stride = 2
        encoder_batch_norm = False  # True
        encoder_num_layers = 4

        # decoder params
        decoder_num_layers = 4
        decoder_filters = [128, 64, 32, 3]
        decoder_kernels = (3, 3)
        decoder_stride = 1
        decoder_batch_norm = False  # [True, True, True, False]
        decoder_activations = [True, True, True, False]
        decoder_upsampling_size = (2, 2)

        # build encoder
        encoder_inp = layers.Input(shape=input_shape, name="encoder_Input")
        label_inp = layers.Input(shape=(dim_y,), name="label_input")
        encoder_deep = model_helper.get_deep_conv(
            "encoder",
            encoder_num_layers,
            encoder_inp,
            encoder_filters,
            encoder_kernels,
            encoder_stride,
            encoder_batch_norm,
        )
        encoder_flat = layers.Flatten(name="encoder_Flatten")(encoder_deep)
        latent_mu = layers.Dense(latent_size, name="latent_mu")(encoder_flat)
        latent_sigma = layers.Dense(latent_size, name="latent_sigma")(encoder_flat)
        z = layers.Lambda(model_helper.sampling, output_shape=(latent_size,), name="z")(
            [latent_mu, latent_sigma]
        )

        encoder = models.Model(
            [encoder_inp, label_inp], [latent_mu, latent_sigma, z], name="encoder"
        )

        # build decoder
        decoder_inp = layers.Input(z.shape[1:], name="decoder_Input")
        decoder_dense = layers.Dense(4096, name="decoder_Dense")(decoder_inp)
        decoder_reshape = layers.Reshape((4, 4, 256), name="decoder_Reshape")(
            decoder_dense
        )

        decoder_outp = model_helper.get_deep_conv_with_upsampling(
            "decoder",
            decoder_num_layers,
            decoder_reshape,
            decoder_upsampling_size,
            decoder_filters,
            decoder_kernels,
            decoder_stride,
            decoder_batch_norm,
            decoder_activations,
        )

        decoder = models.Model(decoder_inp, decoder_outp, name="decoder")

        vae_outp = decoder(encoder([encoder_inp, label_inp])[-1])

        classifier_1 = layers.Dense(latent_size, name="classifier_1")(latent_mu)
        classifier_2 = layers.Dense(latent_size // 2, name="classifier_2")(classifier_1)
        classifier_3 = layers.Dense(
            dim_y, activation="softmax", name="classifier_outp"
        )(classifier_2)

        classifier = models.Model(
            [encoder_inp, label_inp], classifier_3, name="classifier"
        )

        if log_gradients := self.get_log_gradients():
            gradient_file = os.path.join(self.get_model_dir(), "gradient_file")

            if log_gradients == "memory_efficient":
                vae = model_helper.MemoryEfficientGradientModel(
                    [encoder_inp, label_inp],
                    vae_outp,
                    name="vae",
                    gradient_file=gradient_file,
                )
            else:
                vae = model_helper.GradientModel(
                    [encoder_inp, label_inp],
                    vae_outp,
                    name="vae",
                    gradient_file=gradient_file,
                )
        else:
            # TODO handling of multiple outputs everywhere (callbacks, generation, ...)
            # vae = models.Model([encoder_inp, label_inp], [vae_outp, classifier_3], name="vae")
            vae = models.Model([encoder_inp, label_inp], vae_outp, name="vae")

        flat_input = tf.reshape(encoder_inp, (-1, np.prod(input_shape)))
        flat_prediction = tf.reshape(vae_outp, (-1, np.prod(input_shape)))
        # reconstruction_loss = losses.mean_squared_logarithmic_error(
        #     flat_input, flat_prediction
        # ) * np.prod(input_shape)

        reconstruction_loss = losses.mean_squared_error(
            flat_input, flat_prediction
        ) * np.prod(input_shape)

        kl_loss = -0.5 * K.sum(
            1 + latent_sigma - K.square(latent_mu) - K.exp(latent_sigma),
            axis=-1,
        )

        classifier_loss = losses.categorical_crossentropy(label_inp, classifier_3)

        vae_loss = (
            beta * reconstruction_loss + alpha * kl_loss + gamma * classifier_loss
        )
        vae.add_loss(vae_loss if self.get_per_example_loss() else K.mean(vae_loss))

        vae.add_metric(reconstruction_loss, name="reconstruction_loss")
        vae.add_metric(kl_loss, name="kl_loss")
        vae.add_metric(classifier_loss, name="classifier_loss")
        catacc = tf.keras.metrics.categorical_accuracy(label_inp, classifier_3)
        vae.add_metric(catacc, name="classifier_cat_acc")

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.vae = vae

        super().create_model()

    def train_model(self) -> None:
        super().train_model()


class MultitaskBinVAE(VAEContainer):
    def __init__(self, model_config: str, config: dict, verbose: int = 1):
        self._conditional = True
        super().__init__(model_config, config, verbose)

    def create_model(self, trained: bool = False) -> None:

        self.check_if_data_shape_is_present()
        latent_size = self.get_latent_dim()
        input_shape = self._data_shape

        # loss weighting parameters
        alpha = 1  # kl-loss
        beta = 1  # reconstruction-loss
        gamma = 1  # classifier

        # encoder params
        encoder_filters = [32, 64, 128, 256]
        encoder_kernels = (4, 4)
        encoder_stride = 2
        encoder_batch_norm = False  # True
        encoder_num_layers = 4

        # decoder params
        decoder_num_layers = 4
        decoder_filters = [128, 64, 32, 3]
        decoder_kernels = (3, 3)
        decoder_stride = 1
        decoder_batch_norm = False  # [True, True, True, False]
        decoder_activations = [True, True, True, False]
        decoder_upsampling_size = (2, 2)

        # build encoder
        encoder_inp = layers.Input(shape=input_shape, name="encoder_Input")
        label_inp = layers.Input(shape=(1,), name="label_input")
        encoder_deep = model_helper.get_deep_conv(
            "encoder",
            encoder_num_layers,
            encoder_inp,
            encoder_filters,
            encoder_kernels,
            encoder_stride,
            encoder_batch_norm,
        )
        encoder_flat = layers.Flatten(name="encoder_Flatten")(encoder_deep)
        latent_mu = layers.Dense(latent_size, name="latent_mu")(encoder_flat)
        latent_sigma = layers.Dense(latent_size, name="latent_sigma")(encoder_flat)
        z = layers.Lambda(model_helper.sampling, output_shape=(latent_size,), name="z")(
            [latent_mu, latent_sigma]
        )

        encoder = models.Model(
            [encoder_inp, label_inp], [latent_mu, latent_sigma, z], name="encoder"
        )

        # build decoder
        decoder_inp = layers.Input(z.shape[1:], name="decoder_Input")
        decoder_dense = layers.Dense(4096, name="decoder_Dense")(decoder_inp)
        decoder_reshape = layers.Reshape((4, 4, 256), name="decoder_Reshape")(
            decoder_dense
        )

        decoder_outp = model_helper.get_deep_conv_with_upsampling(
            "decoder",
            decoder_num_layers,
            decoder_reshape,
            decoder_upsampling_size,
            decoder_filters,
            decoder_kernels,
            decoder_stride,
            decoder_batch_norm,
            decoder_activations,
        )

        decoder = models.Model(decoder_inp, decoder_outp, name="decoder")

        vae_outp = decoder(encoder([encoder_inp, label_inp])[-1])

        # Classifier
        classifier_1 = layers.Dense(latent_size, name="classifier_1")(latent_mu)
        classifier_2 = layers.Dense(latent_size // 2, name="classifier_2")(classifier_1)

        classifier_3 = layers.Dense(1, activation="sigmoid", name="classifier_outp")(
            classifier_2
        )

        classifier = models.Model(
            [encoder_inp, label_inp], classifier_3, name="classifier"
        )

        if log_gradients := self.get_log_gradients():
            gradient_file = os.path.join(self.get_model_dir(), "gradient_file")

            if log_gradients == "memory_efficient":
                vae = model_helper.MemoryEfficientGradientModel(
                    [encoder_inp, label_inp],
                    vae_outp,
                    name="vae",
                    gradient_file=gradient_file,
                )
            else:
                vae = model_helper.GradientModel(
                    [encoder_inp, label_inp],
                    vae_outp,
                    name="vae",
                    gradient_file=gradient_file,
                )
        else:
            # TODO handling of multiple outputs everywhere (callbacks, generation, ...)
            # vae = models.Model(
            #     [encoder_inp, label_inp], [vae_outp, classifier_3], name="vae"
            # )
            vae = models.Model([encoder_inp, label_inp], vae_outp, name="vae")

        flat_input = tf.reshape(encoder_inp, (-1, np.prod(input_shape)))
        flat_prediction = tf.reshape(vae_outp, (-1, np.prod(input_shape)))

        reconstruction_loss = losses.mean_squared_error(
            flat_input, flat_prediction
        ) * np.prod(input_shape)

        kl_loss = -0.5 * K.sum(
            1 + latent_sigma - K.square(latent_mu) - K.exp(latent_sigma),
            axis=-1,
        )

        classifier_loss = losses.binary_crossentropy(label_inp, classifier_3)

        vae_loss = (
            beta * reconstruction_loss + alpha * kl_loss + gamma * classifier_loss
        )
        vae.add_loss(vae_loss if self.get_per_example_loss() else K.mean(vae_loss))

        vae.add_metric(reconstruction_loss, name="reconstruction_loss")
        vae.add_metric(kl_loss, name="kl_loss")
        vae.add_metric(classifier_loss, name="classifier_loss")
        binacc = tf.keras.metrics.binary_accuracy(label_inp, classifier_3)

        vae.add_metric(binacc, name="classifier_bin_acc")

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.vae = vae

        super().create_model()

    def train_model(self) -> None:
        super().train_model()


class MultitaskSensorDataVAE(VAEContainer):
    def __init__(self, model_config: str, config: dict, verbose: int = 1):
        self._conditional = True
        super().__init__(model_config, config, verbose)

    def create_model(self, trained: bool = False) -> None:
        self.check_if_data_shape_is_present()
        latent_size = self.get_latent_dim()
        input_shape = self._data_shape
        dim_y = self.data.get_num_unique_labels()
        intermediate_dim = self.get_lstm_size()
        lower_bound = self.get_lower_bound()

        # loss weighting parameters
        alpha = 0.01  # kl-loss
        beta = 50  # reconstruction-loss
        gamma = 0.5  # classifier

        # build encoder
        encoder_inp = layers.Input(shape=input_shape, name="encoder_Input")
        label_inp = layers.Input(shape=(dim_y,), name="label_input")
        h = layers.LSTM(intermediate_dim)(encoder_inp)

        if lower_bound is not None and (lower_bound > 0):
            latent_mu = layers.Dense(
                latent_size, name="z_mean", activation=model_helper.custom_scaled_tanh
            )(h)
            latent_sigma = layers.Dense(
                latent_size,
                name="z_log_sigma",
                activation=model_helper.sigma_bound(lower_bound=lower_bound),
            )(h)
        else:
            latent_mu = layers.Dense(latent_size, name="z_mean")(h)
            latent_sigma = layers.Dense(latent_size, name="z_log_sigma")(h)

        z = layers.Lambda(model_helper.sampling, output_shape=(latent_size,), name="z")(
            [latent_mu, latent_sigma]
        )
        encoder = models.Model(
            [encoder_inp, label_inp], [latent_mu, latent_sigma, z], name="encoder"
        )

        # build decoder
        decoder_inp = layers.Input(z.shape[1:], name="decoder_Input")
        r = layers.RepeatVector(input_shape[0])(decoder_inp)
        hidden_lstm_int = layers.LSTM(intermediate_dim, return_sequences=True)(r)
        decoder_outp = layers.LSTM(
            input_shape[1], activation="tanh", return_sequences=True, name="output_lstm"
        )(hidden_lstm_int)

        decoder = models.Model(decoder_inp, decoder_outp, name="decoder")

        # classifier
        classifier_inp = layers.Input(z.shape[1:], name="classifier_Input")
        reshape = layers.Reshape((latent_size, 1), name="classifier_reshape")(
            classifier_inp
        )
        conv1 = layers.Conv1D(
            32, 1, strides=1, activation="relu", name="classifier_conv1"
        )(reshape)
        # dropout1 = layers.Dropout(0.1, name="classifier_dropout1")(conv1)
        dropout1 = conv1
        conv2 = layers.Conv1D(
            64, 2, strides=1, activation="relu", name="classifier_conv2"
        )(dropout1)
        # dropout2 = layers.Dropout(0.1, name="classifier_dropout2")(conv2)
        dropout2 = conv2
        conv3 = layers.Conv1D(96, 1, strides=1, activation="relu")(dropout2)
        # dropout3 = layers.Dropout(0.1, name="classifier_dropout3")(conv3)
        dropout3 = conv3
        globalMaxPool = layers.GlobalMaxPooling1D(name="classifier_maxpool")(dropout3)
        dense1 = layers.Dense(32, activation="sigmoid", name="classifier_dense1")(
            globalMaxPool
        )
        dense2 = layers.Dense(dim_y, activation="softmax", name="classifier_output")(
            dense1
        )
        classifier = models.Model(classifier_inp, dense2, name="classifier")

        vae_outp = decoder(encoder([encoder_inp, label_inp])[-1])
        # change here, if classifier should focus on mu (0), sigma (1) or z (-1)
        classifier_outp = classifier(encoder([encoder_inp, label_inp])[0])

        if log_gradients := self.get_log_gradients():
            gradient_file = os.path.join(self.get_model_dir(), "gradient_file")

            if log_gradients == "memory_efficient":
                vae = model_helper.MemoryEfficientGradientModel(
                    [encoder_inp, label_inp],
                    vae_outp,
                    name="vae",
                    gradient_file=gradient_file,
                )
            else:
                vae = model_helper.GradientModel(
                    [encoder_inp, label_inp],
                    vae_outp,
                    name="vae",
                    gradient_file=gradient_file,
                )
        else:
            # TODO handling of multiple outputs everywhere (callbacks, generation, ...)
            # vae = models.Model([encoder_inp, label_inp], vae_outp, name="vae")
            vae = models.Model([encoder_inp, label_inp], vae_outp, name="vae")

        flat_input = tf.reshape(encoder_inp, (-1, np.prod(input_shape)))
        flat_prediction = tf.reshape(vae_outp, (-1, np.prod(input_shape)))

        reconstruction_loss = losses.mean_squared_error(flat_input, flat_prediction)

        kl_loss = -0.5 * K.mean(
            1 + latent_sigma - K.square(latent_mu) - K.exp(latent_sigma)
        )

        classifier_loss = losses.categorical_crossentropy(label_inp, classifier_outp)

        vae_loss = (
            beta * reconstruction_loss + alpha * kl_loss + gamma * classifier_loss
        )
        vae.add_loss(vae_loss if self.get_per_example_loss() else K.mean(vae_loss))

        if not trained:
            # vae.add_metric(reconstruction_loss, name="reconstruction_loss")
            # vae.add_metric(kl_loss, name="kl_loss")
            vae.add_metric(classifier_loss, name="classifier_loss")
            catacc = tf.keras.metrics.categorical_accuracy(label_inp, classifier_outp)
            vae.add_metric(catacc, name="classifier_cat_acc")

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.vae = vae

        super().create_model()

    def train_model(self) -> None:
        super().train_model()

    # def generate_new_data(
    #     self,
    #     sample_size: int = 1,
    #     labels: list or int = None,
    #     num_classes: int = None,
    # ) -> list:
    #     """This function generates new data. Special handling because decoder has no label information.

    #     Args:
    #         sample_size (int, optional): Number of random new samples to generate. Defaults to 1.
    #         labels (list or int, optional): For conditional models the target labels. Defaults to None.
    #         num_classes (int, deprecated): For compability reasons only.

    #     Raises:
    #         RuntimeError: Model not present.
    #         RuntimeError: Missing label information.

    #     Returns:
    #         list: Newly generated data.
    #     """

    #     if self.vae is None:
    #         raise RuntimeError(
    #             f"{self.__class__} has no model present. Create one first to generate data."
    #         )

    #     if labels is None:
    #         raise RuntimeError(
    #             f"To generate data for {self.__class__} pass label information."
    #         )

    #     dim_z = self.decoder.input_shape[-1]

    #     labels = util.check_if_list_and_matches_length(labels, sample_size, "labels")
    #     global_zs = np.zeros((sample_size, dim_z))

    #     list_of_labels, counts = np.unique(labels, return_counts=True)

    #     for label, count in zip(list_of_labels, counts):
    #         label_zs = []

    #         while True:
    #             tmp_z = np.random.default_rng().normal(size=(count, dim_z))
    #             pred_conf = self.classifier.predict(tmp_z)
    #             pred_val = np.argmax(pred_conf, axis=1)
    #             mask = pred_val == label

    #             if sum(mask) > 0:
    #                 label_zs += list(tmp_z[mask])
    #             else:
    #                 label_zs.append(tmp_z[np.argmax(pred_conf[:, label])])

    #             if len(label_zs) >= count:
    #                 break
    #         label_zs = np.array(label_zs)[:count]
    #         global_zs[labels == label] = label_zs

    #     return self.decoder(global_zs)
