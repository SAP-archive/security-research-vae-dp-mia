import json
import os
from typing import Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, utils

import util.models as mh
from core.dataset import SimpleConditionalDataGenerator


class ClassifierOptimizer:
    def __init__(self, num_calls: int = 20) -> None:

        self.configs_dir: str = "./configs/optimizer_configs"
        self.num_calls = num_calls

        if not os.path.isdir(self.configs_dir):
            os.makedirs(self.configs_dir)

    def save_results(self, results: dict, savepath: str):
        results_json = {
            "learning_rate": float(results.x[0]),
            "batch_size": int(results.x[1]),
            "epochs": int(results.x[2]),
            "val_accuracy": float(1.0 - results.fun),
        }

        with open(savepath, "w+") as fw:
            json.dump(results_json, fw)

    def set_search_space(
        self,
        learning_rate_bound: Tuple[float, float] = (1e-5, 1e-2),
        batch_size_range: Tuple[int, int] = (4, 7),
        epoch_bound: Tuple[int, int] = (10, 50),
    ) -> None:
        self.search_space = [
            Real(*learning_rate_bound, "log_uniform", name="learning_rate"),
            Categorical([2**i for i in range(*batch_size_range)], name="batch_size"),
            Integer(*epoch_bound, name="epochs"),
        ]

    def optimize_vgg16_classifier(
        self,
        num_classes: int,
        eps: int or float or None = None,
        noise: int or float or None = None,
        dataset: str = "lfw-64-64",
    ):
        self.set_search_space()

        data_folder = f"./data/{dataset}-{num_classes}"
        if eps:
            data_file = f"{dataset}-{num_classes}_eps_{eps}_m_64_b_1.npz"
        elif noise:
            data_file = f"{dataset}-{num_classes}_noise_{noise}.npz"
        else:
            data_file = f"{dataset}-{num_classes}.npz"

        data_path = os.path.join(data_folder, data_file)

        with np.load(data_path, allow_pickle=True) as data:
            X, y = data["X"] / 255, data["y"]
            y_cat = utils.to_categorical(y, num_classes=num_classes)

        global objective_function

        @use_named_args(self.search_space)
        def objective_function(learning_rate: float, batch_size: int, epochs: int):

            print(f"Using lr={learning_rate}\tbatch_size={batch_size}\tepochs={epochs}")

            num_train_data: int or float = 0.5
            num_val_data: int or float = 0.2

            first_sss = StratifiedShuffleSplit(n_splits=1, train_size=num_train_data)
            train_idx, remain_idx = next(first_sss.split(X, y))

            train_generator = SimpleConditionalDataGenerator(
                X[train_idx], y_cat[train_idx], batch_size, True
            )

            second_sss = StratifiedShuffleSplit(n_splits=1, train_size=num_val_data)
            val_idx, test_idx = next(second_sss.split(X[remain_idx], y[remain_idx]))

            val_generator = SimpleConditionalDataGenerator(
                X[remain_idx][val_idx],
                y_cat[remain_idx][val_idx],
                batch_size,
                True,
            )
            test_generator = SimpleConditionalDataGenerator(
                X[remain_idx][test_idx],
                y_cat[remain_idx][test_idx],
                batch_size,
                True,
            )

            mdl = mh.customVGG16Model(
                optimizer=optimizers.Adam(learning_rate=learning_rate)
            )
            mdl.create(num_classes=num_classes)
            mdl.fit(
                train_generator,
                epochs=epochs,
                batch_size=batch_size,
                val_generator=val_generator,
            )

            val_acc = mdl.evaluate(test_generator)[-1]
            print(f"Test Accuracy: {val_acc}")
            del mdl
            K.clear_session()

            return 1.0 - val_acc

        res = gp_minimize(
            objective_function, self.search_space, n_calls=30, verbose=True
        )

        savepath = os.path.join(
            self.configs_dir, "vgg16-" + ".".join(data_file.split(".")[:-1]) + ".json"
        )

        self.save_results(res, savepath)

    def optimize_harcnn_classifier(
        self,
        num_classes: int = 6,
        eps: int or float or None = None,
        noise: int or float or None = None,
        dataset: str = "MotionSenseConditional",
    ):

        self.set_search_space(batch_size_range=(5, 10), epoch_bound=(5, 25))

        data_folder = f"./data/{dataset}"
        if eps:
            data_file = f"{dataset}_eps_{eps}.npz"
        elif noise:
            data_file = f"{dataset}_noise_{noise}.npz"
        else:
            data_file = f"{dataset}.npz"

        data_path = os.path.join(data_folder, data_file)

        with np.load(data_path, allow_pickle=True) as data:
            X, y = data["X"], data["y"]

        if not (X.shape[1] == 12 and X.shape[2] == 500):
            X = X.reshape(-1, 12, 500)
            y_cat = utils.to_categorical(y, num_classes=num_classes)

        global objective_function

        @use_named_args(self.search_space)
        def objective_function(learning_rate: float, batch_size: int, epochs: int):

            print(f"Using lr={learning_rate}\tbatch_size={batch_size}\tepochs={epochs}")

            num_train_data: int or float = 0.5
            num_val_data: int or float = 0.2

            first_sss = StratifiedShuffleSplit(n_splits=1, train_size=num_train_data)
            train_idx, remain_idx = next(first_sss.split(X, y))

            train_generator = SimpleConditionalDataGenerator(
                X[train_idx], y_cat[train_idx], batch_size, True
            )

            second_sss = StratifiedShuffleSplit(n_splits=1, train_size=num_val_data)
            val_idx, test_idx = next(second_sss.split(X[remain_idx], y[remain_idx]))

            val_generator = SimpleConditionalDataGenerator(
                X[remain_idx][val_idx],
                y_cat[remain_idx][val_idx],
                batch_size,
                True,
            )
            test_generator = SimpleConditionalDataGenerator(
                X[remain_idx][test_idx],
                y_cat[remain_idx][test_idx],
                batch_size,
                True,
            )

            mdl = mh.customHARCNNModel(
                optimizer=optimizers.Adam(learning_rate=learning_rate)
            )
            mdl.create(num_classes=num_classes)
            mdl.fit(
                train_generator,
                epochs=epochs,
                batch_size=batch_size,
                val_generator=val_generator,
            )

            val_acc = mdl.evaluate(test_generator)[-1]
            print(f"Test Accuracy: {val_acc}")
            del mdl
            K.clear_session()

            return 1.0 - val_acc

        res = gp_minimize(
            objective_function, self.search_space, n_calls=self.num_calls, verbose=True
        )

        savepath = os.path.join(
            self.configs_dir, "harcnn-" + ".".join(data_file.split(".")[:-1]) + ".json"
        )

        self.save_results(res, savepath)


opt = ClassifierOptimizer()

lfw_list_of_classes = [20, 50, 100]
lfw_list_of_eps = [None, 10000, 5000, 1000, 500, 100]
for cl in lfw_list_of_classes:
    for eps in lfw_list_of_eps:
        opt.optimize_vgg16_classifier(num_classes=cl, eps=eps)


ms_list_of_eps = [None, 10, 1, 0.1, 0.01]
for eps in ms_list_of_eps:
    opt.optimize_harcnn_classifier(eps=eps)


list_of_noise = [0.01, 0.1, 1]
for cl in lfw_list_of_classes:
    for noise in list_of_noise:
        opt.optimize_vgg16_classifier(num_classes=cl, noise=noise)

for noise in list_of_noise:
    opt.optimize_harcnn_classifier(noise=noise)
