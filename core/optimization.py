import gc
import json
import os
import shutil
import traceback
from math import inf, isinf, isnan
from typing import Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import util.models as model_helper
import util.utilities as util
from sklearn.model_selection import StratifiedShuffleSplit
from skopt import gp_minimize
from skopt.space import Categorical, Real
from skopt.utils import use_named_args
from tensorflow.keras import optimizers, utils
from tensorflow.keras.backend import clear_session

from core.dataset import (
    BaseImageClass,
    BaseTimeSeriesClass,
    SimpleConditionalDataGenerator,
)
from core.model import ModelContainer


class ModelOptimizer:
    def __init__(
        self,
        model_config_name: str,
        num_gp_calls: int = 20,
        overwrite_config: bool = True,
        save_results_seperately: bool = False,
    ) -> None:
        """Class to run bayesian optimization on model configs.

        Args:
            model_config_name (str): Name of the config to load.
            overwrite_config (bool, optional): Whether to store the results in the config file. Defaults to True.
            save_results_seperately (bool, optional): Whether the results should be stored seperately. Defaults to False.
        """
        self.model_config_name = model_config_name
        self.num_gp_calls = num_gp_calls
        self.overwrite_config = overwrite_config
        self.save_results_seperately = save_results_seperately

    def prepare_optimization(self) -> None:
        """Calls functions to setup basics that all optimization strategies need."""
        self.set_all_paths_and_basics()
        self.set_search_space()

    def _test_if_keys_in_model_config(self, *args) -> bool:
        for key in args:
            if not key in self.model_config.keys():
                return False
        return True

    def _set_path_to_vgg_conf(self, base_path: str) -> None:

        if self._test_if_keys_in_model_config("epsilon", "train", "test", "val"):
            if (
                self.model_config["epsilon"] is not None
                and self.model_config["train"]
                and self.model_config["test"]
                and self.model_config["val"]
            ):
                self.path_to_vgg_conf = os.path.join(
                    base_path,
                    "/".join(ModelContainer._configs_dir.split("/")[:-1]),
                    "optimizer_configs",
                    "_".join(
                        [
                            "vgg16-" + self.model_config["dataset"],
                            "eps",
                            str(self.model_config["epsilon"]),
                            "m",
                            str(self.model_config["m"])
                            if self._test_if_keys_in_model_config("m")
                            else "64",
                            "b",
                            str(self.model_config["b"])
                            if self._test_if_keys_in_model_config("b")
                            else "1",
                        ]
                    )
                    + ".json",
                )
                return

        self.path_to_vgg_conf = os.path.join(
            base_path,
            "/".join(ModelContainer._configs_dir.split("/")[:-1]),
            "optimizer_configs",
            ".".join(["vgg16-" + self.model_config["dataset"], "json"]),
        )

    def _set_path_to_harcnn_conf(self, base_path: str):
        if self._test_if_keys_in_model_config("epsilon", "train", "test", "val"):
            if (
                self.model_config["epsilon"] is not None
                and self.model_config["train"]
                and self.model_config["test"]
                and self.model_config["val"]
            ):
                self.path_to_harcnn_conf = os.path.join(
                    base_path,
                    "/".join(ModelContainer._configs_dir.split("/")[:-1]),
                    "optimizer_configs",
                    f"harcnn-{self.model_config['dataset']}_eps_{self.model_config['epsilon']}.json",
                )
                return

        self.path_to_harcnn_conf = os.path.join(
            base_path,
            "/".join(ModelContainer._configs_dir.split("/")[:-1]),
            "optimizer_configs",
            f"harcnn-{self.model_config['dataset']}.json",
        )

    def set_all_paths_and_basics(self) -> None:
        """Builds the base paths for the optimization target model, the tmp files and the vgg."""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.path_to_orig_config = os.path.join(
            base_path,
            ModelContainer._configs_dir,
            ".".join([self.model_config_name, "json"]),
        )

        self.path_to_tmp_config = os.path.join(
            base_path,
            ModelContainer._configs_dir,
            ".".join(["_".join(["tmp", self.model_config_name]), "json"]),
        )

        self.path_to_orig_model_dir = os.path.join(
            base_path, ModelContainer._models_dir, self.model_config_name
        )

        self.path_to_tmp_model_dir = os.path.join(
            base_path,
            ModelContainer._models_dir,
            "_".join(["tmp", self.model_config_name]),
        )

        with open(self.path_to_orig_config, "r") as f:
            self.model_config = json.load(f)

        self._set_path_to_vgg_conf(base_path)
        self._set_path_to_harcnn_conf(base_path)

    def set_search_space(
        self,
        learning_rate_bound: Tuple[float, float] = (1e-5, 1e-2),
        batch_size_range: Tuple[int, int] = (4, 7),
    ) -> None:
        """Sets the search space for the optimization"""
        self.search_space = [
            Real(*learning_rate_bound, "log_uniform", name="learning_rate"),
            Categorical([2**i for i in range(*batch_size_range)], name="batch_size"),
        ]

    def flush_adjusted_orig_config(self, res) -> None:
        """Checks if the original config should be overriden and overrides with the best params if so

        Args:
            res (dict): results from gp_minimize call
        """

        if self.overwrite_config:
            self.model_config["batch_size"] = int(res.x[1])
            self.model_config["optimizer_params"]["learning_rate"] = float(res.x[0])

            tmp_modulo = (
                self.model_config["epoch_modulo"]
                if self._test_if_keys_in_model_config("epoch_modulo")
                else False
            )
            tmp_chkpt = (
                self.model_config["checkpoint_model"]
                if self._test_if_keys_in_model_config("checkpoint_model")
                else False
            )
            tmp_gradients = (
                self.model_config["log_gradients"]
                if self._test_if_keys_in_model_config("log_gradients")
                else False
            )

            self.model_config = ModelOptimizer.check_tmp_config(self.model_config)

            self.model_config["epoch_modulo"] = tmp_modulo
            self.model_config["checkpoint_model"] = tmp_chkpt
            self.model_config["log_gradients"] = tmp_gradients

            with open(self.path_to_orig_config, "w") as f:
                json.dump(self.model_config, f, cls=util.CustomJSONEncoder)

    def save_results(self, res):
        """Checks if the results should be saved seperately and saves them to the model folder if so.

        Args:
            res (dict): results from gp_minimize call
        """
        if self.save_results_seperately:
            json_res = {"batch_size": int(res.x[1]), "learning_rate": float(res.x[0])}

            if not os.path.isdir(self.path_to_orig_model_dir):
                os.makedirs(self.path_to_orig_model_dir)

            with open(self.path_to_orig_model_dir, "w+") as f:
                json.dump(json_res, f, cls=util.CustomJSONEncoder)

    def cleanup_optimization(self):
        """Removes the temporary files and folders that get created during optimization."""
        if os.path.isfile(self.path_to_tmp_config):
            os.remove(self.path_to_tmp_config)

        if os.path.isdir(self.path_to_tmp_model_dir):
            shutil.rmtree(self.path_to_tmp_model_dir)

    @staticmethod
    def check_tmp_config(tmp_config: dir) -> dir:
        """Function to allow custom checks on the tmp_config"""

        if "num_microbatches" in tmp_config["optimizer_params"].keys():
            # batch_size = tmp_config["batch_size"]
            # if batch_size % 4 == 0:
            #     tmp_config["optimizer_params"]["num_microbatches"] = batch_size // 4
            # else:
            #     tmp_config["optimizer_params"]["num_microbatches"] = batch_size // 2
            tmp_config["optimizer_params"]["num_microbatches"] = 4

        # disable analysis flags
        tmp_config["log_gradients"] = False
        tmp_config["checkpoint_model"] = False
        tmp_config["epoch_modulo"] = False

        return tmp_config

    def optimize(self) -> None:
        """Uses gp_minimize from sklearn to optimize the given model due to test loss."""

        path_to_tmp_config = self.path_to_tmp_config
        model_config_name = self.model_config_name
        config = self.model_config.copy()

        @use_named_args(self.search_space)
        def objective_function(learning_rate: float, batch_size: int):
            try:
                print(
                    "learning_rate", learning_rate, "batch_size", batch_size, sep="\t"
                )
                tmp_config = config.copy()

                tmp_config["batch_size"] = int(batch_size)
                tmp_config["optimizer_params"]["learning_rate"] = float(learning_rate)

                tmp_config = ModelOptimizer.check_tmp_config(tmp_config)

                with open(path_to_tmp_config, "w+") as f:
                    json.dump(tmp_config, f, cls=util.CustomJSONEncoder)

                mdl = ModelContainer.create(
                    "_".join(["tmp", model_config_name]), verbose=1
                )
                mdl.load_data()

                mdl.create_model()
                mdl.train_model()
                loss = mdl.evaluate_loss(save=False)["test_loss"]

                mdl.clear_model()

                del mdl

            except Exception as e:
                print(e)
                traceback.print_exc()
                loss = inf

            # reduce tf internal state size s.t. we do not run out of memory
            clear_session()
            gc.collect()

            # upper bound for nan / inf, randomness s.t. not always the same value is used
            loss_upper_bound = float(np.random.randint(1, 10) * 10e10)

            if isnan(loss) or isinf(loss) or loss > loss_upper_bound:
                # loss = np.finfo("float32").max / 1e4
                loss = loss_upper_bound
            else:
                loss = float(loss)

            return loss

        res = gp_minimize(
            objective_function,
            self.search_space,
            n_calls=self.num_gp_calls,
            verbose=True,
        )

        self.flush_adjusted_orig_config(res)
        self.save_results(res)
        self.cleanup_optimization()

    def optimize_as_data_generator_for_vgg(self) -> None:
        """
            We use model to generate train data for a classifier. The closer the model gets to the baseline accuracy on original data, the better the generated data is.

        Raises:
            RuntimeError: When no conditional information is present.
            RuntimeError: When config file for vgg optimizer is missing.
        """

        path_to_tmp_config = self.path_to_tmp_config
        model_config_name = self.model_config_name
        config = self.model_config.copy()

        # load data and train vgg as baseline on real data
        mdl = ModelContainer.create(self.model_config_name, verbose=0)
        mdl.load_data()

        if not isinstance(mdl.data, BaseImageClass):
            print("optimize_as_data_generator_for_vgg expects image data. Skipped.")
            return

        x_train, x_test, _, y_train, y_test, _ = mdl.data.unravel()

        if y_train is None or y_test is None:
            raise RuntimeError(
                "Optimization for VGG only possible with conditional information."
            )

        num_classes = mdl.data.get_num_unique_labels()

        if not os.path.isfile(self.path_to_vgg_conf):
            raise RuntimeError(
                f"Need a config for vgg classifier, but {self.path_to_vgg_conf} not present."
            )

        with open(self.path_to_vgg_conf, "r") as f:
            vgg_config = json.load(f)

        vgg_learning_rate, vgg_batch_size, vgg_epochs = (
            vgg_config["learning_rate"],
            vgg_config["batch_size"],
            vgg_config["epochs"],
        )

        del vgg_config

        inp_shape = x_train.shape
        num_training_samples = inp_shape[0]

        train_generator = SimpleConditionalDataGenerator(
            x_train, y_train, vgg_batch_size, True
        )

        # ensure that there are not many more samples in test than in train
        if len(x_test) > num_training_samples + 2 * num_classes:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=num_training_samples)

            idx, _ = next(sss.split(x_test, y_test))
        else:
            idx = np.arange(len(x_test))

        test_generator = SimpleConditionalDataGenerator(
            x_test[idx], y_test[idx], vgg_batch_size, True
        )

        del mdl, x_train, x_test, y_test

        vgg16 = model_helper.customVGG16Model(
            optimizer=optimizers.Adam(vgg_learning_rate)
        )
        vgg16.create(input_shape=inp_shape[1:], num_classes=num_classes)
        vgg16.fit(train_generator, epochs=vgg_epochs, batch_size=vgg_batch_size)
        baseline_test_accuracy = vgg16.evaluate(test_generator)[-1]

        print(f"Baseline Test Accuracy is: {baseline_test_accuracy}")

        # keep the information about the distribution of the train data to generate similar datasets

        vgg_train_labels = np.argmax(y_train, axis=1)

        del train_generator, vgg16
        clear_session()
        gc.collect()

        @use_named_args(self.search_space)
        def objective_function(learning_rate: float, batch_size: int):
            try:
                print(
                    "learning_rate", learning_rate, "batch_size", batch_size, sep="\t"
                )
                tmp_config = config.copy()

                tmp_config["batch_size"] = int(batch_size)
                tmp_config["optimizer_params"]["learning_rate"] = float(learning_rate)

                tmp_config = ModelOptimizer.check_tmp_config(tmp_config)

                with open(path_to_tmp_config, "w+") as f:
                    json.dump(tmp_config, f, cls=util.CustomJSONEncoder)

                mdl = ModelContainer.create(
                    "_".join(["tmp", model_config_name]), verbose=1
                )
                mdl.load_data()
                mdl.create_model()
                mdl.train_model()

                # generate new data
                gen_train_data = mdl.generate_new_data(
                    num_training_samples,
                    labels=vgg_train_labels,
                    num_classes=num_classes,
                )
                mdl.clear_model()
                del mdl

                train_generator = SimpleConditionalDataGenerator(
                    gen_train_data,
                    utils.to_categorical(vgg_train_labels, num_classes),
                    vgg_batch_size,
                    True,
                )
                # finetune model on generated data
                vgg16 = model_helper.customVGG16Model(
                    optimizer=optimizers.Adam(vgg_learning_rate)
                )
                vgg16.create(input_shape=inp_shape[1:], num_classes=num_classes)
                vgg16.fit(train_generator, epochs=vgg_epochs, batch_size=vgg_batch_size)
                test_accuracy = vgg16.evaluate(test_generator)[-1]
                del vgg16, train_generator

                val_acc_dist = np.abs(baseline_test_accuracy - test_accuracy)

            except Exception as e:
                print(e)
                traceback.print_exc()
                val_acc_dist = inf

            # reduce tf internal state size s.t. we do not run out of memory
            clear_session()
            gc.collect()

            # upper bound for nan / inf, randomness s.t. not always the same value is used
            val_acc_dist_upper_bound = float(np.random.randint(1, 10) * 10e10)

            if (
                isnan(val_acc_dist)
                or isinf(val_acc_dist)
                or val_acc_dist > val_acc_dist_upper_bound
                or val_acc_dist == 0  # assume that model didn't learn anything
            ):
                # val_acc_dist = np.finfo("float32").max / 1e4
                val_acc_dist = val_acc_dist_upper_bound
            else:
                val_acc_dist = float(val_acc_dist)

            return val_acc_dist

        res = gp_minimize(
            objective_function,
            self.search_space,
            n_calls=self.num_gp_calls,
            verbose=True,
        )

        self.flush_adjusted_orig_config(res)
        self.save_results(res)
        self.cleanup_optimization()

    def optimize_as_data_generator_for_harcnn(self) -> None:
        """
            We use model to generate train data for a classifier. The closer the model gets to the baseline accuracy on original data, the better the generated data is.

        Raises:
            RuntimeError: When no conditional information is present.
            RuntimeError: When config file for harcnn optimizer is missing.
        """

        self.set_search_space(batch_size_range=(5, 10))
        path_to_tmp_config = self.path_to_tmp_config
        model_config_name = self.model_config_name
        config = self.model_config.copy()

        # load data and train harcnn as baseline on real data
        mdl = ModelContainer.create(self.model_config_name, verbose=0)
        mdl.load_data()

        if not isinstance(mdl.data, BaseTimeSeriesClass):
            print(
                "optimize_as_data_generator_for_harcnn expects time series data. Skipped."
            )
            return

        x_train, x_test, _, y_train, y_test, _ = mdl.data.unravel()

        if y_train is None or y_test is None:
            raise RuntimeError(
                "Optimization for harcnn only possible with conditional information."
            )

        num_classes = mdl.data.get_num_unique_labels()

        if not os.path.isfile(self.path_to_harcnn_conf):
            raise RuntimeError(
                f"Need a config for harcnn classifier, but {self.path_to_harcnn_conf} not present."
            )

        with open(self.path_to_harcnn_conf, "r") as f:
            harcnn_config = json.load(f)

        harcnn_learning_rate, harcnn_batch_size, harcnn_epochs = (
            harcnn_config["learning_rate"],
            harcnn_config["batch_size"],
            harcnn_config["epochs"],
        )

        del harcnn_config

        inp_shape = x_train.shape
        num_training_samples = inp_shape[0]

        train_generator = SimpleConditionalDataGenerator(
            x_train, y_train, harcnn_batch_size, True
        )

        # ensure that there are not many more samples in test than in train
        if len(x_test) > num_training_samples + 2 * num_classes:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=num_training_samples)

            idx, _ = next(sss.split(x_test, y_test))
        else:
            idx = np.arange(len(x_test))

        test_generator = SimpleConditionalDataGenerator(
            x_test[idx], y_test[idx], harcnn_batch_size, True
        )

        del mdl, x_train, x_test, y_test

        harcnn = model_helper.customHARCNNModel(
            optimizer=optimizers.Adam(harcnn_learning_rate)
        )
        harcnn.create(input_shape=inp_shape[1:], num_classes=num_classes)
        harcnn.fit(train_generator, epochs=harcnn_epochs, batch_size=harcnn_batch_size)
        baseline_test_accuracy = harcnn.evaluate(test_generator)[-1]

        print(f"Baseline Test Accuracy is: {baseline_test_accuracy}")

        # keep the information about the distribution of the train data to generate similar datasets

        harcnn_train_labels = np.argmax(y_train, axis=1)

        del train_generator, harcnn
        clear_session()
        gc.collect()

        @use_named_args(self.search_space)
        def objective_function(learning_rate: float, batch_size: int):
            try:
                print(
                    "learning_rate", learning_rate, "batch_size", batch_size, sep="\t"
                )
                tmp_config = config.copy()

                tmp_config["batch_size"] = int(batch_size)
                tmp_config["optimizer_params"]["learning_rate"] = float(learning_rate)

                tmp_config = ModelOptimizer.check_tmp_config(tmp_config)

                with open(path_to_tmp_config, "w+") as f:
                    json.dump(tmp_config, f, cls=util.CustomJSONEncoder)

                mdl = ModelContainer.create(
                    "_".join(["tmp", model_config_name]), verbose=1
                )
                mdl.load_data()
                mdl.create_model()
                mdl.train_model()
                clear_session()
                # generate new data
                gen_train_data = mdl.generate_new_data(
                    num_training_samples,
                    labels=harcnn_train_labels,
                    num_classes=num_classes,
                )
                mdl.clear_model()
                del mdl

                train_generator = SimpleConditionalDataGenerator(
                    gen_train_data,
                    utils.to_categorical(harcnn_train_labels, num_classes),
                    harcnn_batch_size,
                    True,
                )
                # finetune model on generated data
                harcnn = model_helper.customHARCNNModel(
                    optimizer=optimizers.Adam(harcnn_learning_rate)
                )
                harcnn.create(input_shape=inp_shape[1:], num_classes=num_classes)
                harcnn.fit(
                    train_generator, epochs=harcnn_epochs, batch_size=harcnn_batch_size
                )
                test_accuracy = harcnn.evaluate(test_generator)[-1]
                del harcnn, train_generator

                val_acc_dist = np.abs(baseline_test_accuracy - test_accuracy)

            except Exception as e:
                print(e)
                traceback.print_exc()
                val_acc_dist = inf

            # reduce tf internal state size s.t. we do not run out of memory
            clear_session()
            gc.collect()

            # upper bound for nan / inf, randomness s.t. not always the same value is used
            val_acc_dist_upper_bound = float(np.random.randint(1, 10) * 10e10)

            if (
                isnan(val_acc_dist)
                or isinf(val_acc_dist)
                or val_acc_dist > val_acc_dist_upper_bound
                or val_acc_dist == 0  # assume that model didn't learn anything
            ):
                # val_acc_dist = np.finfo("float32").max / 1e4
                val_acc_dist = val_acc_dist_upper_bound
            else:
                val_acc_dist = float(val_acc_dist)

            return val_acc_dist

        res = gp_minimize(
            objective_function,
            self.search_space,
            n_calls=self.num_gp_calls,
            verbose=True,
        )

        self.flush_adjusted_orig_config(res)
        self.save_results(res)
        self.cleanup_optimization()
