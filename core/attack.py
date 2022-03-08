import json
import os
from typing import Iterable, List, Union

import cv2
import numpy as np
import scipy
import tensorflow as tf
import util.figures as figure_helper
import util.lpips_tf as lpips_tf
import util.utilities as util
from skimage.feature import hog
from skimage.metrics import structural_similarity as ssim
from sklearn import metrics
from sklearn.decomposition import PCA
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import to_categorical

from core.base import BaseClass
from core.model import ModelContainer

"""Log Levels ('verbose' parameter)

0 -- silent
1 -- minimal log output
2 -- detailed log output

"""


class Attacker(BaseClass):

    _configs_dir: str = "configs/attack_configs"

    # ------------------------------------- Basic methods -------------------------------------

    def __init__(
        self,
        model_config_name: str,
        verbose: int = 1,
    ):
        """Construct an attacker object around a given model.

        Args:
            model_config_name (str): Name of the config of the model to attack.
            verbose (int, optional): Verbosity level. Is used for logging. Defaults to 0.

        Raises:
            Exception: If the supplied attack config is not valid.
        """

        self._verbose = verbose  # controls log output

        self._print_log: bool = False
        self._write_log: bool = True

        # Can be used to do repetitive tasks
        self._util = util.BaseHelper(self)
        self._type = "attacker"

        # Folder that contains all scripts and sub folders
        self._base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self._model_config_name = model_config_name
        self._model_present_: bool = False

    def _get_attack_type(self) -> str:
        return self._util.read_config("attack_type")

    def _get_ensure_orig_data(self) -> bool:
        return self._util.read_config("ensure_orig_data", strict=False) is True

    def set_attack_config(self, attack_config_name: str, attack_config: dict = None):
        """Set the next attack config that should be used.

        Args:
            attack_config_name (str): Name of the attack config to use.
            attack_config (dict, optional): Instead of the config name the actual config dict can be provided. Defaults to None.

        Raises:
            Exception: [description]
        """
        if attack_config is None:
            # Load config JSON from filesystem
            with open(
                os.path.join(
                    self._base_path, self._configs_dir, attack_config_name + ".json"
                ),
                "r",
            ) as f:
                attack_config = json.loads(f.read())

        reason = self._util.config_valid(attack_config)
        if reason is not None:
            raise Exception("Supplied config not valid, reason: {0}".format(reason))

        self.config_name_ = attack_config_name
        self._config = attack_config

    def _load_model(self, use_checkpoint: bool or int = False):
        """Creates a reference model and loads its data

        Uses the in the attack config supplied model config name to
        create a model (helper). Calls the load data method to acquire
        the required training, validation and test data.

        Args.
            use_checkpoint(bool or int, optional): Whether to use the model from a specific checkpoint. Defaults to False.
        """

        clear_session()

        model_config_name = self._model_config_name
        # Creates either GAN or VAE model
        # WARNING: It is assumed that the model was already trained prior to the import/attack
        self._model = ModelContainer.create(model_config_name, verbose=self._verbose)

        self._model._config["force_new_data_indices"] = False

        # load model needs info from data
        self._model.load_data()

        if use_checkpoint:
            self._model.create_model(trained=True)
            self._model.load_checkpoint_at(use_checkpoint)
        else:
            self._model.load_model()

        self._model_present_ = True

    def prepare_attack(self, force: bool = False, use_checkpoint: bool or int = False):
        """Executes necessary preparation steps before the actual attack

        Args.
            force (bool, optional): Whether to force model load. Defaults to False.
            use_checkpoint (bool or int, optional): Whether to use the model from a specific checkpoint. Defaults to False.
        """

        self._use_checkpoint = use_checkpoint

        if (not self._model_present_) or force or use_checkpoint:
            self._load_model(use_checkpoint)

        model_name = self._model._model_name
        self._util.log(f"Attacking model: {model_name}", 1)

    def perform_attack(self, subset: np.array = None, attack_sample_size: int = 1000):
        """Perform the actual attack as specified in attack config

        Args:
            subset (np.array, optional): A list of indices, specifying which samples to use in an attack.
            attack_sample_size (int, optional): Defines how many samples should be used for attack measurements. Is ignored when subset is specified. Defaults to 100.

        Raises:
            Exception: If differing amount of training and test samples are supplied.
            Exception: If image samples do not have quadratic shapes.
            Exception: If sample size is invalid (must be multiple of the amount of labels).
            Exception: If an unknown attack type is specified in config.
        """

        # Seed is set globally in script
        # Set random seed one time before attack(s)
        # if self._random_seed_ is not None:
        #    np.random.seed(self._random_seed_)

        attack_type = self._get_attack_type()
        ensure_orig_data = self._get_ensure_orig_data()

        self._util.log(f"Performing '{attack_type}' on model", 1)

        data_ref = None
        if ensure_orig_data:

            data_conf = self._model.data._data_conf.copy()
            (
                data_conf["train"],
                data_conf["test"],
                data_conf["val"],
                data_conf["force_new_data_indices"],
            ) = (False, False, False, False)

            data_dir, dataset = self._model._util.read_config("data_dir", "dataset")
            perturbation_conf = {"data_ldp_noise": False, "epsilon": False}

            data_ref = self._model.data_cls(
                self._model._util,
                os.path.join(self._model._base_path, data_dir),
                dataset,
                data_conf,
                perturbation_conf,
                self._model.data.get_data_indices(),
            )
            data_ref.preprocess_data("VAE")

        else:
            data_ref = self._model.data

        # Collect data
        if subset is None:
            # Read data from model and reduce training and test to attack_sample_size
            x_train, x_test, _, y_train, y_test, _ = data_ref.unravel(
                limit=attack_sample_size, random_order=True
            )

        else:
            # Use given index to pick subset as attack data
            x_train, x_test, _, y_train, y_test, _ = data_ref.unravel()
            x_train = x_train[subset]
            x_test = x_test[subset]
            y_train = y_train[subset]
            y_test = y_test[subset]

        x_train, x_test, y_train, y_test = self._ensure_equal_length(
            x_train, x_test, y_train, y_test, attack_sample_size
        )

        # Validate attack configuration
        if "category" in attack_type:
            sample_size = self._util.read_config("sample_size")
            label_cnt = self._mode._data.get_num_unique_labels()
            if sample_size % label_cnt != 0:
                raise Exception(
                    "Amount of samples must be multiple of amount of labels (e.g. 10 labels and 100 Samples)"
                )

        data_params = {
            "train_data": x_train,
            "test_data": x_test,
            "train_labels": y_train,
            "test_labels": y_test,
        }

        # ! FIXME only reconstruction & taxonomy attack tested on all models
        # Trigger actual attack
        if attack_type == "wb_attack":
            self._wb_attack(**data_params)
        elif attack_type == "pca_mc_attack":
            self._pca_mc_attack(**data_params, category_attack=False)
        elif attack_type == "pca_mc_category_attack":
            self._pca_mc_attack(**data_params, category_attack=True)
        elif attack_type == "hog_mc_attack":
            self._hog_mc_attack(**data_params, category_attack=False)
        elif attack_type == "hog_mc_category_attack":
            self._hog_mc_attack(**data_params, category_attack=True)
        elif attack_type == "chist_mc_attack":
            self._color_hist_attack(**data_params, category_attack=False)
        elif attack_type == "chist_mc_category_attack":
            self._color_hist_attack(**data_params, category_attack=True)
        elif attack_type == "reconstruction_attack":
            self._reconstruction_attack(**data_params)
        elif attack_type == "taxonomy_attack":
            # TODO overhead because of lpips implementation. move to tf2
            tf.compat.v1.disable_eager_execution()
            self.prepare_attack(force=True, use_checkpoint=self._use_checkpoint)
            self._taxonomy_attack(**data_params)

        else:
            raise ValueError(f"Unknown attack type {attack_type}")

        self._save_result()

    def _save_result(self) -> None:

        save_path = self._model.get_attack_results_dir()
        save_file = os.path.join(save_path, "attack_results.json")

        if os.path.isfile(save_file):

            with open(save_file, "r") as f:
                prev_results = json.load(f)
        else:
            prev_results = dict()

        prev_results[self.config_name_] = self.results

        with open(save_file, "w") as f:
            json.dump(prev_results, f, cls=util.CustomJSONEncoder)

    def _log_maybe(self, text: str, new: bool = False) -> None:
        """Helper function for writing/printing logs

        Args:
            text (str): Text to write.
            new (bool, optional): Indicates start of log (file). Defaults to False.
        """

        if self._write_log:

            log_path = os.path.join(self._base_path, "logs")
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            log_file_name = self.config_name_ + "_log.txt"
            file_path = os.path.join(log_path, log_file_name)

            if not new:
                mode = "a"
            else:
                mode = "w"
                self._util.log(f"{self.__class__} writing new log to `{file_path}`", 1)

            with open(file_path, mode) as log:
                log.write(text)

        if self._print_log:
            self._util.log(text, 0)

    def _write_attack_log(self, data: dict) -> None:
        """Writes attack results

        Depending on value of _write_log and _print_log results can
        get printed and/or written to log file.

        Args:
            data (dict): Dictionary containing the attack results.
        """

        attack_type = self._get_attack_type()

        text = "\n###### Attack Results ######\n"
        self._log_maybe(text, new=True)

        text = f"Config: {self.config_name_}\nAttack Type: {attack_type}\nResults:\n"
        self._log_maybe(text)

        if "mc_attack" in attack_type or "mc_category_attack" in attack_type:
            for percentile in data.keys():
                text = f"Percentile: {percentile}\n"
                self._log_maybe(text)

                for attack in data[percentile].keys():
                    text = f"\t{attack}: {data[percentile][attack]}\n"
                    self._log_maybe(text)

        elif "reconstruction_attack" in attack_type:
            text = f"\tAccuracy: {data['reconstruction_accuracy']}\n\tSuccessful Set Attack: {data['successful_set_attack']}\n"
            self._log_maybe(text)

        # if self._write_log:
        #     log_file_name = self.config_name_ + "_log.txt"
        #     self._util.log("Results written to log file: {0}".format(log_file_name), 1)

    def _full_attack_evaluation(
        self, true: list, conf: list, pred: list, error_mode: bool = False
    ) -> dict:
        """[summary]

        Args:
            true (list): List of true labels.
            conf (list): List of confidence values, i.e., calculated distances or errors. Is normalized between 0 and 1.
            pred (list): List of predictions inferred from the conf.
            error_mode (bool, optional): Whether a high conf determines in (True) or out (False). For example, former happens for ssim, latter for mse. Defaults to False.

        Returns:
            dict: [description]
        """

        save_dir = os.path.join(self._model.get_attack_results_dir(), "attack_figures")

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        true = true.astype(int)
        pred = pred.astype(int)

        conf = (conf - min(conf)) / (max(conf) - min(conf))
        if error_mode:
            conf = np.abs(conf - 1)

        prec, rec, _ = metrics.precision_recall_curve(true, conf)
        ap = metrics.average_precision_score(true, conf)

        figure_helper.plot_curve(
            rec,
            prec,
            f"AP={ap:.2f}",
            "recall",
            "precision",
            [[0, 1], [0.5, 0.5]],
            os.path.join(
                save_dir, f"{self.config_name_.lower()}_precison_recall_curve.pdf"
            ),
        )

        fpr, tpr, tresh = metrics.roc_curve(true, conf)
        auc = metrics.roc_auc_score(true, conf)

        figure_helper.plot_curve(
            fpr,
            tpr,
            f"AUC={auc:.2f}",
            "specificity",
            "sensitivity",
            [[0, 1], [0, 1]],
            os.path.join(save_dir, f"{self.config_name_.lower()}_roc_curve.pdf"),
        )

        adv = np.clip(tpr - fpr, a_min=0, a_max=None)
        adv_auc = metrics.auc(tresh, adv)

        figure_helper.plot_curve(
            tresh,
            adv,
            f"ADV_AUC={adv_auc:.2f}",
            "threshold",
            "advantage",
            [[0, 1], [0, 0]],
            os.path.join(save_dir, f"{self.config_name_.lower()}_advantage_curve.pdf"),
        )

        cm = metrics.confusion_matrix(true, pred)

        figure_helper.plot_confusion_matrix(
            cm,
            [0, 1],
            savepath=os.path.join(
                save_dir, f"{self.config_name_.lower()}_confusion_matrix.pdf"
            ),
        )
        figure_helper.plot_confusion_matrix(
            cm,
            [0, 1],
            True,
            savepath=os.path.join(
                save_dir, f"{self.config_name_.lower()}_confusion_matrix_normalized.pdf"
            ),
        )

        self.results["prec_rec_curve"] = {"prec": prec, "rec": rec, "ap": ap}
        self.results["roc_curve"] = {"fpr": fpr, "tpr": tpr, "auc": auc}
        self.results["adv_curve"] = {"tresh": tresh, "adv": adv, "adv_auc": adv_auc}
        self.results["confusion_matrix"] = cm

    def print_attack_details(self) -> None:
        """Prints a list of attack details"""

        attack_type, sample_size = self._get_attack_type(), self._util.read_config(
            "sample_size"
        )
        model_type, dataset = self._model._util.read_config("type", "dataset")

        self._util.log("\n###### Attack Details ######\n", 0)
        self._util.log(f"Attack config: {self.config_name_}", 0)
        self._util.log(f"Attacked model: {self.model_config_name}", 0)
        self._util.log(f"Model type: {model_type}", 0)
        self._util.log(f"Attack type: {attack_type}", 0)
        self._util.log(f"Used data set: {dataset}\n", 0)
        self._util.log(f"Sample Size: {sample_size}\n", 0)

    def _ensure_equal_length(
        self,
        X_one: np.array,
        X_two: np.array,
        y_one: np.array = None,
        y_two: np.array = None,
        size: int = None,
    ) -> tuple:
        """Takes two arrays and returns them equally sized.

        Args:
            X_one (np.array): First array to check.
            X_two (np.array): Second array to check.
            y_one (np.array, optional): Optional labels for the first array. Defaults to None.
            y_two (np.array, optional): Optional labels for the first array. Defaults to None.
            size (int, optional): Optional a predetermined maximal size of the arrays. When None, the smaller array determines size. Defaults to None.

        Returns:
            tuple: Arrays of equal length.
        """
        len_one, len_two = X_one.shape[0], X_two.shape[0]

        if size is None:
            size = min(len_one, len_two)

        if size > len_one or size > len_two:
            size = min(len_one, len_two)

        return (
            X_one[:size],
            X_two[:size],
            y_one[:size] if y_one is not None else None,
            y_two[:size] if y_two is not None else None,
        )

    # ------------------------------------- WB attack -------------------------------------

    def _wb_attack(
        self,
        train_data: np.array,
        test_data: np.array,
        train_labels: np.array,
        test_labels: np.array,
    ) -> None:
        """White box discriminator attack

        Code taken from 'wb_attack'

        Args:
            train_data (np.array): The samples used for training of the model.
            test_data (np.array): Samples not seen during training.
            train_labels (np.array): The labels of the training samples.
            test_labels (np.array): The Labels of the unseen samples
        """

        if (model_type := self._model._type) != "GAN":
            raise ValueError(
                f"Whitebox discriminator attack can only be done for GAN models (instance class is {model_type})"
            )

        # Container for training images results ("1" indicator)
        results_train = np.ones((len(train_labels), 2))
        # Container for validation images results ("0" indicator)
        results_test = np.zeros((len(test_labels), 2))

        # Get discriminator values
        results_train[:, 1] = np.squeeze(self._discriminate(train_data, train_labels))
        results_test[:, 1] = np.squeeze(self._discriminate(test_data, test_labels))

        # Combine, sort according to confidence
        results = np.concatenate((results_train, results_test))
        results = results[results[:, 1].argsort()]

        accuracy = results[-len(results_train) :, 0].mean()
        successful_set_attack = bool(sum(results_train[:, 1]) > sum(results_test[:, 1]))

        pred = np.zeros((len(results), 1), dtype=int)
        pred[-len(results_train) :] = 1

        self.results = {
            "whitebox_accuracy": accuracy,
            "successful_set_attack": successful_set_attack,
        }
        self._write_attack_log(self.results)
        self._full_attack_evaluation(results[:, 0], results[:, 1], pred)

    def _discriminate(self, X: np.array, y: np.array) -> np.array:
        """Retrieve discriminator values for given samples

        Args:
            X (np.array): Samples to feed in discriminator.
            y (np.array): Corresponding lables.

        Returns:
            np.array: Discrimanator value for each sample.
        """
        return self._model.discriminator.predict([X, y])

    # ------------------------------------- MC attacks -------------------------------------

    def _generate_samples(
        self,
        sample_size: int,
        labels: Iterable[int],
        return_labels: bool = False,
        random_labels: bool = False,
    ) -> Union[np.array, List[np.array]]:
        """Uses generator to create new samples

        Args:
            sample_size (int): Amount of samples to generate.
            labels (Iterable[int]): Unique list of class labels.
            return_labels (bool, optional): True if labels of samples should be returned. Defaults to False.
            random_labels (bool, optional): True if labels of samples should be drawn randomly. Defaults to False.

        Returns:
            Union[np.array, List[np.array]]: Either generated samples or samples and corresponding labels.
        """

        #! FIXME doesnt work for non-conditional models yet!
        rng = np.random.default_rng()

        if random_labels:
            # Draw random labels from list
            drawn_labels = rng.choice(labels, size=sample_size)
        else:
            # Every label equally frequent
            drawn_labels = np.repeat(labels, sample_size // len(labels))

        # Create random z according to model configuration
        dim_z, model_type = self._model._util.read_config("dim_z", "type")

        mean = np.ones(dim_z)
        cov = np.eye(dim_z)
        z = rng().multivariate_normal(mean, cov, sample_size)

        # Generate
        if model_type == "GAN":
            # One-hot encode
            # ATTENTION: Model concatenates z (noise) and y (condition) -> dim_y has to be the same as dim_z
            dim_y = dim_z
            y = to_categorical(drawn_labels, dim_y)

            generated_samples = self._model.generator.predict([z, y])

        elif model_type == "VAE":
            # One-hot encode (with vector size = n_labels)
            y = to_categorical(drawn_labels)

            conc = np.concatenate([z, y], axis=1)
            generated_samples = self._model.decoder.predict(conc)

            # Reshape into image size
            # Only works with quadratic images
            sample_shape = (len(generated_samples),) + self._model._data_shape
            generated_samples = np.reshape(generated_samples, sample_shape)

            if not util.has_quadratic_shape(generated_samples):
                raise Exception(
                    'Quadratic shape is expected, but shape of data ist "{0}"'.format(
                        generated_samples.shape
                    )
                )

        else:
            raise Exception('Unknown model type "{0}"'.format(model_type))

        # Shift from [-1,1] to [0,1]?
        generated_samples = (generated_samples + 1.0) / 2.0

        # Shift from [0,1] to [0,255]?
        generated_samples = generated_samples - generated_samples.min()
        generated_samples = generated_samples * 255 / generated_samples.max()

        if return_labels:
            return [generated_samples, drawn_labels]
        else:
            return generated_samples

    def _calculate_mc_metrics(
        self, distances: np.array, epsilon: int, train: bool
    ) -> np.array:
        """Calculates result metric values for given distances

        Args:
            distances (np.array): Distances either between train and generated or validation and generated data.
            epsilon (int): Calculated threshold (according to chosen percentile).
            train (bool): Boolean to distinguish training from validation data (for evaluation).

        Returns:
            np.array: The metrics for each sample.
        """

        results = np.empty(len(distances), dtype=dict)

        for i in range(len(results)):

            # Metrics
            integral_approx_log = 0
            integral_approx_eps = 0
            integral_approx_frac = 0

            for distance in distances[i]:
                # Only consider those samples that are within the eps environment
                # of the generated samples
                if distance < epsilon:
                    integral_approx_log += -np.log(
                        distance / epsilon
                    )  # Part of equation 2
                    integral_approx_eps += 1  # Part of equation 1
                    integral_approx_frac += epsilon / distance

            # How large are the distances between our sample and the generated
            # samples that are within the eps-environment
            # Compare equation 2 in Monte Carlo paper
            integral_approx_log = integral_approx_log / len(distances[0])

            # How many generated samples are within the eps-environment of current sample
            # Compare equation 1 in Monte Carlo paper
            integral_approx_eps = integral_approx_eps / len(distances[0])

            # Integral approx frac (not mentioned in paper?)
            integral_approx_frac = integral_approx_frac / len(distances[0])

            results[i] = {
                "integral_approx_log": integral_approx_log,
                "integral_eps_log": integral_approx_eps,
                "integral_approx_frac": integral_approx_frac,
                "train": train,  # Indicate if dataset is a training sample
            }

        return results

    def _evaluate_mc_results(
        self, results_train: np.array, results_test: np.array
    ) -> dict:
        """Compares train and validation results and calculates overall result metrics

        Args:
            results_train (np.array): List of metric values for distances between generated and train data.
            results_test (np.array): List of metric values for distances between generated and validation data.

        Returns:
            dict: Dictionary with the overall results according to the different criterias.

        """

        all_results = np.concatenate((results_train, results_test))
        np.random.default_rng().shuffle(all_results)

        values = dict()

        # For each metric use precalculated values (see _calculate_mc_metrics) to produce
        # mean metric values of the training and validation samples and calculate an
        # accuracy (how likely are we to detect training samples)
        for abbreviation, metric in [
            ("log", "integral_approx_log"),
            ("eps", "integral_eps_log"),
            ("frac", "integral_approx_frac"),
        ]:

            # Collect (train and validation values of given metric)
            train_values = [entry[metric] for entry in results_train]
            test_values = [entry[metric] for entry in results_test]

            # Single membership inference
            # Produce mean
            mean_train = np.mean(train_values)
            mean_test = np.mean(test_values)

            # Sort according to metric values and select top half (highest values)
            all_results_sorted = sorted(all_results, key=lambda item: item[metric])
            top_half = all_results_sorted[-len(results_train) :]

            # Count amount of true (train) entries in top half and divide by the amount of total entries (mean)
            single_attack_acc = sum([entry["train"] for entry in top_half]) / len(
                results_train
            )

            # Set membership inference
            # Compare sum of training and validation values of metric to
            # decide if set attack was successful
            successful_set_attack = bool(sum(train_values) > sum(test_values))

            # Save results
            values["mc_attack_" + abbreviation + "_train_mean"] = mean_train
            values["mc_attack_" + abbreviation + "_test_mean"] = mean_test
            values["mc_attack_" + abbreviation + "_acc"] = single_attack_acc
            values["successful_set_attack_" + abbreviation] = successful_set_attack

        return values

    def _monte_carlo_attack(
        self,
        mc_type: str,
        train_data: np.array,
        train_labels: np.array,
        test_data: np.array,
        test_labels: np.array,
        category_attack: bool,
    ) -> None:
        """Method to perform Monte Carlo attacks (PCA, HOG and CHIST)

        Generates "sample_size" samples

        Args:
            mc_type (str): 'pca', 'hog' or 'chist'.
            train_data (np.array): The samples used for training of the model.
            train_labels (np.array): The labels of the training samples.
            test_data (np.array): Samples not seen during training.
            test_labels (np.array): The labels of the unseen samples.
            category_attack (bool): If True, labels are respected during attack.

        """

        # Parameters
        repetitions, sample_size = self._util.read_config("repetitions", "sample_size")

        # Find set of labels
        labels = self._model.data.get_unique_labels()

        # Split data into batches
        if category_attack:
            # Only distances between samples of same class are measured
            # sample_size // len(labels) entries per batch are required
            distances_train = np.zeros(
                (len(train_data), repetitions * sample_size // len(labels))
            )
            distances_test = np.zeros(
                (len(test_data), repetitions * sample_size // len(labels))
            )

        else:
            # Provide space for all combinations between generated samples
            # and train/val samples for each batch (repetition)
            distances_train = np.zeros((len(train_data), repetitions * sample_size))
            distances_test = np.zeros((len(test_data), repetitions * sample_size))

        for batch_no in range(repetitions):

            self._util.log("Batch {0}/{1}".format(batch_no + 1, repetitions), 1)

            # TODO: Sample test/val data here again? (Or work with same data in every repetition)

            if category_attack:
                generated_samples, sample_labels = self._generate_samples(
                    sample_size, labels, return_labels=True
                )
            else:
                generated_samples = self._generate_samples(
                    sample_size, labels, return_labels=False, random_labels=True
                )

            # Perform attack specific transform before distance measurement
            if mc_type == "hog":
                # Transform
                generated_samples = self._generate_batch_hog_features(generated_samples)

            elif mc_type == "pca":
                # Flatten (only if model output has more than two dimensions)
                if generated_samples.ndim >= 3:
                    sample_shape = (
                        len(generated_samples),
                        np.prod(generated_samples.shape[1:]),
                    )
                    generated_samples = np.reshape(generated_samples, sample_shape)

                # Transform
                generated_samples = self._pca.transform(generated_samples)

            elif mc_type == "chist":
                # Transform
                generated_samples = self._calculate_batch_hist(generated_samples)

            if category_attack:

                for label in labels:
                    # Find indexes of samples with current label
                    indexes_train = np.where(train_labels == label)[0]
                    indexes_test = np.where(test_labels == label)[0]
                    indexes_sample = np.where(sample_labels == label)[0]

                    # Make sure that dimensions fit
                    if len(indexes_sample) != (sample_size // len(labels)):
                        raise Exception(
                            "Error in category attack: Only {0} samples with label {1} when it should have been {2}".format(
                                len(indexes_sample), label, (sample_size // len(labels))
                            )
                        )

                    start = batch_no * (sample_size // len(labels))
                    end = (batch_no + 1) * (sample_size // len(labels))

                    # Calculate distances between samples and train data (with current label)
                    distances_train[
                        indexes_train, start:end
                    ] = scipy.spatial.distance.cdist(
                        train_data[indexes_train],
                        generated_samples[indexes_sample],
                        "euclidean",
                    )
                    distances_test[
                        indexes_test, start:end
                    ] = scipy.spatial.distance.cdist(
                        test_data[indexes_test],
                        generated_samples[indexes_sample],
                        "euclidean",
                    )

            else:

                start = batch_no * sample_size
                end = (batch_no + 1) * sample_size

                # Calculate distances between samples and train data
                distances_train[:, start:end] = scipy.spatial.distance.cdist(
                    train_data, generated_samples, "euclidean"
                )
                distances_test[:, start:end] = scipy.spatial.distance.cdist(
                    test_data, generated_samples, "euclidean"
                )

        heuristics = self._util.read_config("heuristics")
        # heuristics.append('median')

        results = dict()

        # Perform evaluation for every heuristic (described in paper)
        for heuristic in heuristics:

            # Choose epsilon for the epsilon environment (threshold)
            if heuristic == "median":
                distances = np.concatenate((distances_train, distances_test))
                # epsilon = np.median([distances[i].min() for i in range(len(distances))])
                epsilon = np.median(distances.min(axis=1))
            else:
                percentile = float(heuristic)
                epsilon = np.percentile(
                    np.concatenate((distances_train, distances_test)), percentile
                )

            # Calculate metric values (according to paper) for different subsets
            metrics_train = self._calculate_mc_metrics(
                distances_train, epsilon, train=True
            )
            metrics_test = self._calculate_mc_metrics(
                distances_test, epsilon, train=False
            )

            # Compare train and validation results and calculate result metrics
            evaluation = self._evaluate_mc_results(metrics_train, metrics_test)
            evaluation["eps"] = epsilon
            results[heuristic] = evaluation

        self._util.log(results, required_level=2)

        # results = self._evaluate_distances(distances_train, distances_test, percentiles, median_heuristic=True)
        self.results = results
        self._write_attack_log(results)

    # ------------------------------------- PCA MC attack -------------------------------------

    def _pca_mc_attack(
        self,
        train_data: np.array,
        test_data: np.array,
        train_labels: np.array,
        test_labels: np.array,
        category_attack: bool = False,
    ):
        """PCA based Monte Carlo attack

        Code taken from 'euclidean_pca_mc_attack_category'

        Args:
            train_data (np.array): The samples used for training of the model.
            train_labels (np.array): The labels of the training samples.
            test_data (np.array): Samples not seen during training.
            test_labels (np.array): The labels of the unseen samples.
            category_attack (bool, optional): If True distances are calculated per label. Defaults to False.

        """

        # Parameters
        n_components_pca = self._util.read_config("n_components_pca")

        # Load test data to train pca
        test_data = self._model.data.X_test

        # Check data shape
        # if len(train_data.shape) > 3 and train_data.shape[3] > 1:
        #    raise Exception('PCA MC Attack does not work with sample shape "{0}"'.format(train_data.shape))

        # Reverse one-hot encoding
        train_labels = train_labels.argmax(axis=1)
        test_labels = test_labels.argmax(axis=1)

        # Flatten image data (preparation for PCA)
        if test_data.ndim >= 3:
            prod = np.prod(test_data.shape[1:])  # 784 (mnist/fmnist) or 3072 (cifar10)
            test_data = np.reshape(test_data, (len(test_data), prod))
            train_data = np.reshape(train_data, (len(train_data), prod))
            test_data = np.reshape(test_data, (len(test_data), prod))

        # Create and fit PCA
        pca = PCA(n_components=n_components_pca)
        pca.fit(test_data)
        self._pca = pca

        # Transform data
        train_data = pca.transform(train_data)
        test_data = pca.transform(test_data)

        # Perform actual attack
        self._monte_carlo_attack(
            "pca", train_data, train_labels, test_data, test_labels, category_attack
        )

    # ------------------------------------- HOG MC attack -------------------------------------

    def _generate_batch_hog_features(self, samples: np.array) -> np.array:
        """Produces HOG Feature Vectors for images

        Args:
            samples (np.array): Array of images.

        Returns:
            np.array: Array of HOG feature vectors.

        """

        # Read shape of samples used for model construction/training
        sample_shape = self._model._data_shape

        if samples.shape[1:] != sample_shape:
            raise Exception(
                "Shape of samples {0} does not match required shape {1}".format(
                    samples.shape, sample_shape
                )
            )

        if not util.has_quadratic_shape(samples):
            raise Exception(
                'Quadratic shape is expected, but shape of data ist "{0}"'.format(
                    samples.shape
                )
            )

        # Calculate number of features of hog vector
        cell_size = (9, 9)  # Number of pixels per cell (width, height)
        block_size = (3, 3)  # Number of cells in one block (width, height)
        img_width = sample_shape[0]
        img_height = sample_shape[1]
        orientations = 9  # Number of orientation bins (per cell)

        n_cells_per_block = block_size[0] * block_size[1]
        n_values_per_block = n_cells_per_block * orientations
        n_shifts_horizontal = (img_width - (block_size[0] * cell_size[0])) // cell_size[
            0
        ]
        n_shifts_vertical = (img_height - (block_size[1] * cell_size[1])) // cell_size[
            1
        ]
        n_block_calculations_horizontal = n_shifts_horizontal + 1
        n_block_calculations_vertical = n_shifts_vertical + 1
        n_features = (
            n_block_calculations_horizontal * n_block_calculations_vertical
        ) * n_values_per_block

        # Create container with calculated size
        features_matrix = np.zeros((len(samples), n_features))

        for i in range(len(samples)):
            features_matrix[i] = hog(
                samples[i],
                orientations=orientations,
                pixels_per_cell=cell_size,
                cells_per_block=block_size,
                visualize=False,
            )

        return features_matrix

    def _hog_mc_attack(
        self,
        train_data: np.array,
        test_data: np.array,
        train_labels: np.array,
        test_labels: np.array,
        category_attack: bool = False,
    ):
        """HOG based Monte Carlo attack

        Code taken from 'hog_mc_attack_category' method

        Args:
            train_data (np.array): The samples used for training of the model.
            train_labels (np.array): The labels of the training samples.
            test_data (np.array): Samples not seen during training.
            test_labels (np.array): The labels of the unseen samples.
            category_attack (bool, optional): If True distances are calculated per label. Defaults to False.

        """

        # Check data shape
        if len(train_data.shape) > 3 and train_data.shape[3] > 1:
            raise Exception(
                'HOG MC Attack does not work with sample shape "{0}"'.format(
                    train_data.shape
                )
            )

        # Reverse one-hot encoding
        train_labels = train_labels.argmax(axis=1)
        test_labels = test_labels.argmax(axis=1)

        # Create HOG features
        feature_matrix_train = self._generate_batch_hog_features(train_data)
        feature_matrix_test = self._generate_batch_hog_features(test_data)

        self._monte_carlo_attack(
            "hog",
            feature_matrix_train,
            train_labels,
            feature_matrix_test,
            test_labels,
            category_attack,
        )

    # ------------------------------------- CHIST MC attack -------------------------------------

    def _calculate_hist(self, image: np.array, bins: int) -> np.array:
        """Calculate color histogram for one image.

        Args:
            image (np.array): The image for which the histogram is calculated.
            bins (int): The amount of bins to be used for every color.

        Returns:
            np.array: The histogram as a flattened array.

        """

        channels = image.shape[2]  # 3
        channels_list = list(range(channels))  # [0, 1, 2]
        bins_list = [bins] * channels  # [16, 16, 16]
        ranges = [0, 256] * channels  # [0, 256, 0, 256, 0, 256]

        # Necessary?
        vMin = np.amin(image)
        vMax = np.amax(image)
        image = (image - vMin) / (vMax - vMin) * 255

        hist = cv2.calcHist([np.float32(image)], channels_list, None, bins_list, ranges)
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _calculate_batch_hist(self, images: np.array) -> np.array:

        bins = 16
        v_size = bins ** images.shape[3]  # 16 ** 3 = 4096
        features = np.zeros((len(images), v_size))

        for i in range(len(images)):
            features[i, :] = self._calculate_hist(images[i], bins)

        return features

    def _color_hist_attack(
        self,
        train_data: np.array,
        test_data: np.array,
        train_labels: np.array,
        test_labels: np.array,
        category_attack: bool = False,
    ) -> None:
        """Color Histogram based MC Attack

        Args:
            train_data (np.array): The samples used for training of the model.
            train_labels (np.array): The labels of the training samples.
            test_data (np.array): Samples not seen during training.
            test_labels (np.array): The labels of the unseen samples.
            category_attack (bool, optional): If True distances are calculated per label. Defaults to False.

        """

        # Reverse one-hot encoding
        train_labels = train_labels.argmax(axis=1)
        test_labels = test_labels.argmax(axis=1)

        feature_matrix_train = self._calculate_batch_hist(train_data)
        feature_matrix_test = self._calculate_batch_hist(test_data)

        self._monte_carlo_attack(
            "chist",
            feature_matrix_train,
            train_labels,
            feature_matrix_test,
            test_labels,
            category_attack,
        )

    # ------------------------------------- Reconstruction attack -------------------------------------

    # def _compute_error_small(self, x: np.array, y: np.array, repetitions: int, iterations: int = 3) -> np.array:
    #     """Compute the mean reconstruction error of given samples

    #     Args:
    #         x (np.array): The samples.
    #         y (np.array): corresponding label.
    #         repetitions (int): Defines, how often samples should be repeated during one iteration.
    #         iterations (int, optional): Defines the amount of iterations. Defaults to 3.

    #     Returns:
    #         np.array: A list of mean squared reconstruction errors of x.

    #     """

    #     # Repeat all samples n times
    #     x_repeated = np.repeat(x, repetitions, axis=0)
    #     y_repeated = np.repeat(y, repetitions, axis=0)

    #     # Contains sum of all mses for each sample
    #     mse_sums = np.zeros((len(x)))

    #     for _ in range(iterations):

    #         # Predict for all samples (and all repetitions)
    #         # x_repeated = x_repeated.astype('float32') / 255.
    #         # x_repeated = x_repeated.reshape((len(x_repeated), np.prod(x_repeated.shape[1:])))
    #         x_pred = self._model.vae.predict([x_repeated, y_repeated])

    #         # Pack all results of one sample in one package
    #         packages = np.split(x_pred, len(x))

    #         mse_list = list()

    #         # Iterate over samples
    #         for i in range(len(x)):
    #             # Calculate mse for all repetitions of current sample
    #             # and produce mean over all axes
    #             mse = np.power((x[i] - packages[i]), 2).mean(axis=None)
    #             mse_list.append(mse)

    #         # Element wise addition of mses
    #         mse_sums += np.array(mse_list)

    #     # Produce mean mse over all iterations (for each sample)
    #     return mse_sums / iterations

    def _compute_error(
        self,
        x: np.array,
        y: np.array,
        repetitions: int,
        iterations: int = 3,
        metric: str = "mse",
    ) -> np.array:
        """Compute the mean reconstruction error of given samples

        Args:
            x (np.array): The samples.
            y (np.array): corresponding label.
            repetitions (int): Defines, how often samples should be repeated during one iteration.
            iterations (int, optional): Defines the amount of iterations. Defaults to 3.
            metric (str, optional): Which metric to use to calculate reconstruction error. Defaults to MSE.

        Returns:
            np.array: A list of mean squared reconstruction errors of x.

        """
        # Contains sum of all errors for each sample
        value_sums = np.zeros((len(x)))

        ssim_multichannel = False
        if len(x.shape) > 3:
            ssim_multichannel = True

        for _ in range(iterations):

            # List of mse (current iteration) for each sample
            values = list()

            for idx in range(len(x)):
                xi = x[idx]
                x_repeated = np.repeat([xi], repetitions, axis=0)

                # Predict reconstruction for all repetitions of current sample
                if (
                    self._model._config["dataset"] == "cifar10"
                    and "na" in self._model._model_name
                ):
                    # x_pred = self._model.vae.predict(x_repeated)
                    error = self._model.vae.evaluate(x_repeated)
                    values.append(error)

                else:
                    if self._model._conditional:
                        y_repeated = np.repeat([y[idx]], repetitions, axis=0)
                        x_pred = self._model.vae.predict([x_repeated, y_repeated])
                    else:
                        x_pred = self._model.vae.predict(x_repeated)

                    # Calculate error/similarity for all repetitions of current sample
                    # and produce mean over all axes
                    if metric == "mse":
                        error = np.power((xi - x_pred), 2)

                    elif metric == "ssim":
                        error = np.fromiter(
                            (
                                ssim(xi, xp, multichannel=ssim_multichannel)
                                for xp in x_pred
                            ),
                            dtype=np.float16,
                            count=repetitions,
                        )
                    else:
                        raise ValueError(f"compute error got unknown metric {metric}")

                    values.append(np.mean(error))

            # Element wise addition of mses
            value_sums += np.array(values)

        # Produce mean mse over all iterations (for each sample)
        return value_sums / iterations

    def _reconstruction_attack(
        self,
        train_data: np.array,
        test_data: np.array,
        train_labels: np.array,
        test_labels: np.array,
    ) -> None:
        """Perform reconstruction attack on model

        An amount of "sample_size" training and non-training samples and corresponding
        labels are drawn from data pool. The samples get repeated "repetitions" times
        and fed in the autoencoder. A mean squared error between the original data and
        the reconstructions is calculated for each sample (as a mean of all repetitions).
        It is then calculated how many training samples are in the top 50 percent with
        the lowest reconstruction errors

        Args:
            train_data (np.array): The samples used for training of the model.
            train_labels (np.array): The labels of the training samples.
            test_data (np.array): Samples not seen during training.
            test_labels (np.array): The labels of the unseen samples.

        """

        if self._model._type != "VAE":
            raise ValueError(
                f"Reconstruction attack can only be done for VAE models (instance class is {self._model._type})"
            )

        repetitions, metric = self._util.read_config("repetitions", "metric")

        # Calculate distances for validation samples. First column holds indicator, second holds distances.
        results_sample = np.zeros((len(test_data), 2))

        # Compute mean mse for every sample (using multiple repetitions)
        # try:
        #    results_sample[:, 1] = self._compute_error_small(test_data, test_labels, repetitions)
        # except MemoryError:
        results_sample[:, 1] = self._compute_error(
            test_data, test_labels, repetitions, metric=metric
        )

        # Calculate distances for train samples. First column holds indicator, second holds distances.
        results_train = np.ones((len(train_data), 2))

        # Compute mean mse for every sample (using multiple repetitions)
        # try:
        #     results_train[:, 1] = self._compute_error_small(train_data, train_labels, repetitions)
        # except MemoryError:
        results_train[:, 1] = self._compute_error(
            train_data, train_labels, repetitions, metric=metric
        )

        results = np.concatenate((results_sample, results_train))
        results = results[results[:, 1].argsort()]

        # Sort according to mse (lowest first), select the indicator of first half
        # (assumed that amount of train/val is balanced) and produce mean

        if metric == "mse":
            accuracy = results[:, 0][: len(results_train)].mean()
            successful_set_attack = bool(
                sum(results_train[:, 1]) < sum(results_sample[:, 1])
            )  # compare sum of errors

            pred = np.zeros((len(results), 1), dtype=int)
            pred[: len(results_train)] = 1

        elif metric == "ssim":
            # Reverse
            accuracy = results[:, 0][-len(results_train) :].mean()
            successful_set_attack = bool(
                sum(results_train[:, 1]) > sum(results_sample[:, 1])
            )  # compare sum of similarities

            pred = np.zeros((len(results), 1), dtype=int)
            pred[-len(results_train) :] = 1

        else:
            raise ValueError(f"Reconstruction attack got unknown metric {metric}")

        # accuracy = 1 - results[results[:, 1].argsort()][:, 0][-len(results_train):].mean()
        # print(accuracy)

        self.results = {
            "reconstruction_accuracy": accuracy,
            "successful_set_attack": successful_set_attack,
        }
        self._write_attack_log(self.results)
        self._full_attack_evaluation(
            results[:, 0], results[:, 1], pred, error_mode=(metric == "mse")
        )

    # ------------------------------------- Taxonomy attack -------------------------------------
    def _compute_taxonomy_error(
        self,
        x: np.array,
        y: np.array,
        repetitions: int,
        lambda1: float,
        lambda2: float,
        lambda3: float,
        dim_z: int,
        iterations: int = 1,
    ) -> np.array:
        """Compute the mean reconstruction error of given samples
            L(x, G(z)) = lambda1 * l2(x, G(z)) + lambda2 * L_lpips(x, G(z)) + lambda3 * L_reg(z)
            where
            l2(x, G(z)) = ||x- G(z)||^2
            L_reg(z) = (||z||^2 - dim(z))^2

        Args:
            x (np.array): The samples.
            y (np.array): corresponding label.
            repetitions (int): Defines, how often samples should be repeated during one iteration.
            iterations (int, optional): Defines the amount of iterations. Defaults to 1.

        Returns:
            np.array: A list of distances of x.

        """
        # Contains sum of all errors for each sample
        value_sums = np.zeros((len(x)))

        for _ in range(iterations):

            x_repeated = np.repeat(x, repetitions, axis=0)
            x_shape = x_repeated.shape

            if y is not None:
                y_repeated = np.repeat(y, repetitions, axis=0)
                x_hat = self._model.vae.predict([x_repeated, y_repeated])
                _, _, z = self._model.encoder.predict([x_repeated, y_repeated])
            else:
                x_hat = self._model.vae.predict(x_repeated)
                _, _, z = self._model.encoder.predict(x_repeated)

            # L2
            if lambda1 > 0:
                loss_l2 = tf.reduce_mean(
                    tf.square(
                        x_hat.reshape((x_shape[0], -1))
                        - x_repeated.reshape((x_shape[0], -1))
                    ),
                    axis=[1],
                )
            else:
                loss_l2 = 0

            # L_pips
            if lambda2 > 0:
                loss_lpips = lpips_tf.lpips(x_hat, x_repeated)
            else:
                loss_lpips = 0

            # L_reg
            if lambda3 > 0:
                norm = tf.reduce_sum(tf.square(z), axis=1)
                loss_regularization = (norm - dim_z) ** 2
            else:
                loss_regularization = 0

            # L(x, G(z))
            l_x_g_z = (
                lambda1 * loss_l2 + lambda2 * loss_lpips + lambda3 * loss_regularization
            )

            l_x_g_z = tf.keras.backend.get_value(l_x_g_z)

            value_sums += self._tumbling_window(l_x_g_z, repetitions)

        return value_sums / iterations

    def _tumbling_window(self, x: np.array, interval: int) -> np.array:
        """Slide non-overlapping window over 1d array and compute mean of each window"""
        values = list()

        for i in range(int(len(x) / interval)):
            if i == 0:
                values.append(x[i : (i + 1) * interval].mean())
            else:
                values.append(x[i * interval + 1 : (i + 1) * interval].mean())

        return np.array(values)

    def _taxonomy_attack(
        self,
        train_data: np.array,
        test_data: np.array,
        train_labels: np.array,
        test_labels: np.array,
    ) -> None:
        """
        Perform taxonomy attack on model
        """
        dim_z, model_type = self._model._util.read_config("dim_z", "type")

        if (
            model_type == "PlainPictureVAE"
            or model_type == "ConditionalPlainPictureVAE"
            or model_type == "MultitaskSensorDataVAE"
        ):
            lambda1, lambda2, lambda3, repetitions = self._util.read_config(
                "lambda1", "lambda2", "lambda3", "repetitions"
            )

            # Check test data, first column is indicator, second holds error
            results_sample = np.zeros((len(test_data), 2))
            print("Compute taxonomy distance for test data")
            results_sample[:, 1] = self._compute_taxonomy_error(
                test_data, test_labels, repetitions, lambda1, lambda2, lambda3, dim_z
            )

            # Check train data, first column is indicator, second holds error
            results_train = np.ones((len(train_data), 2))
            print("Compute taxonomy distance for train data")
            results_train[:, 1] = self._compute_taxonomy_error(
                train_data, train_labels, repetitions, lambda1, lambda2, lambda3, dim_z
            )

            results = np.concatenate((results_sample, results_train))
            results = results[results[:, 1].argsort()]

            # Compute accuracy
            accuracy = results[:, 0][: len(results_train)].mean()
            successful_set_attack = bool(
                sum(results_train[:, 1]) < sum(results_sample[:, 1])
            )  # compare sum of errors

            pred = np.zeros((len(results), 1), dtype=int)
            pred[: len(results_train)] = 1

            self.results = {
                "taxonomy_accuracy": accuracy,
                "successful_set_attack": successful_set_attack,
            }

            self._write_attack_log(self.results)
            self._full_attack_evaluation(results[:, 0], results[:, 1], pred, True)

        else:
            raise Exception(
                f"Can not perform taxonomy attack on modeltype `{model_type}`"
            )
