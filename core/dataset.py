import glob
import hashlib
import os
import os.path
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.random import multivariate_normal, normal
from PIL import Image
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.datasets import cifar10, fashion_mnist, mnist
from tensorflow.keras.utils import Sequence, to_categorical
from tqdm import tqdm
from tqdm.contrib import tenumerate
from tqdm.contrib.concurrent import process_map
from util import figures as figure_helper
from util import utilities as util

from core.base import BaseClass


class BaseGenerator(Sequence):
    def __init__(
        self, X: np.ndarray, y: np.ndarray, batch_size: int, to_fit: bool = True
    ) -> None:
        """BaseClass for other generators. __init__ handles basic setup.

        Args:
            X (np.ndarray): Data to create batches from.
            y (np.ndarray): Labels for data.
            batch_size (int): Batch size to use.
            to_fit (bool, optional): Whether to include label information. Meaning depends on subclass. Defaults to True.
        """

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.to_fit = to_fit

    def __len__(self) -> int:
        """Returns the length of the generator, i.e., the number of batches or number of steps.

        Returns:
            int: Number of batches.

        """
        return int(np.ceil(len(self.X) / self.batch_size))

    @abstractmethod
    def __getitem__(self, num: int) -> object:
        pass


class SimpleDataGenerator(BaseGenerator):
    """Class to generate data for generative training, i.e., no target is returned."""

    def __getitem__(self, num: int) -> list:
        """Creates the current batch and returns it. To fit includes or excludes label information for input.

        Args:
            num (int): Number of the batch to return

        Returns:
            list: Data either [X,] or [X,y] for the current batch.
        """

        if self.y is not None and self.to_fit:
            return [
                self.X[num * self.batch_size : (num + 1) * self.batch_size],
                self.y[num * self.batch_size : (num + 1) * self.batch_size],
            ]

        return [
            self.X[num * self.batch_size : (num + 1) * self.batch_size],
        ]


class SimpleConditionalDataGenerator(BaseGenerator):
    """Class to generate data for classification training, i.e., returns label information as target."""

    def __getitem__(self, num: int) -> tuple:
        """Creates the current batch and returns it. To fit includes or excludes target information, i.e., for fit vs. for predict.

        Args:
            num (int): Number of the batch to return

        Returns:
            list: Data either (X,) or (X,y) for the current batch.
        """

        if self.to_fit:
            return (
                self.X[num * self.batch_size : (num + 1) * self.batch_size],
                self.y[num * self.batch_size : (num + 1) * self.batch_size],
            )

        return (self.X[num * self.batch_size : (num + 1) * self.batch_size],)


class DataContainer(BaseClass, ABC):
    """Baseclass for datasets. Determines shared logic for data handling."""

    def __init__(
        self,
        util_ref: object,
        data_path: str,
        data_conf: dict,
        eps_data_path: str = None,
        data_indices: dict = None,
    ) -> None:
        """Initialize for data container. Loads and sets data.

        Args:
            util_ref (object): Reference to BaseHelper, necessary for logging.
            data_path (str): Path to load data from.
            data_conf (dict): Config for the data.
            eps_data_path (str, optional): Second Path to load data from. Sued when config is set to true. Supposed to represent LDP data. Defaults to None.
            data_indices (dict, optional): Indices to recreate the same splits across multiple runs. Defaults to None.
        """
        self._util = util_ref
        self._data_path = data_path
        self._data_conf = data_conf
        self._eps_data_path = eps_data_path
        self._data_indices = data_indices

        self._load_data()

    def _set_data(
        self,
        X_train: np.array,
        X_test: np.array,
        X_val: np.array,
        y_train: np.array,
        y_test: np.array,
        y_val: np.array,
    ) -> None:
        """Set the data of a DataContainer, e.g., after loading or preprocessing.

        Args:
            X_train (np.array): Records supposed for training.
            X_test (np.array): Records supposed for testing.
            X_val (np.array): Records supposed for validation.
            y_train (np.array): Corresponding labels for training data.
            y_val (np.array): Corresponding labels for validation data.
            y_test (np.array): Corresponding labels for test data.

        """

        self._util.log(f"{self.__class__} sets its own data.", required_level=2)

        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

    def get_hash(self) -> str:
        """Create unique, reproducible hash from data stored in container. Can be used to ensure the same split across multiple runs.

        Returns:
            str: Hash generated from the loaded data.
        """
        try:
            all_data = np.vstack([self.X_train, self.X_test, self.X_val])
        except Exception:
            # in rare cases the shape only match after preprocessing ....
            shape = np.prod(self.X_train.shape[1:])
            all_data = np.vstack(
                [
                    self.X_train.reshape((-1, shape)),
                    self.X_test.reshape((-1, shape)),
                    self.X_val.reshape((-1, shape)),
                ]
            )
        copy = all_data.copy(order="C")
        return hashlib.sha1(copy).hexdigest()

    def get_data_indices(self) -> Dict[str, np.array]:
        """Return the used data indices.

        Returns:
            Dict[str, np.array]: Dictionary with indices for train, test and validation.
        """
        return self._data_indices

    def unravel(self, limit: int = None, random_order: bool = False) -> List[np.array]:
        """Unravel data of DataContainer into a list of numpy arrays.

        Args:
            limit (int, optional): Maximum number of records to include in each array. Defaults to None.
            random_order (bool, optional): Shuffle arrays before returning them. Defaults to False.

        Returns:
            List[np.array]: List of data arrays, [X_train, X_test, X_val, y_train, y_test, y_val]
        """

        if random_order:
            rng = np.random.default_rng()
            idx = rng.permutation(len(self.X_train))
            X_train_s = self.X_train[idx][:limit]
            y_train_s = self.y_train[idx][:limit] if self.y_train is not None else None

            idx = rng.permutation(len(self.X_test))
            X_test_s = self.X_test[idx][:limit]
            y_test_s = self.y_test[idx][:limit] if self.y_test is not None else None

            idx = rng.permutation(len(self.X_val))
            X_val_s = self.X_val[idx][:limit]
            y_val_s = self.y_val[idx][:limit] if self.y_val is not None else None

            return [X_train_s, X_test_s, X_val_s, y_train_s, y_test_s, y_val_s]

        else:
            return [
                self.X_train[:limit],
                self.X_test[:limit],
                self.X_val[:limit],
                self.y_train[:limit] if self.y_train is not None else None,
                self.y_test[:limit] if self.y_test is not None else None,
                self.y_val[:limit] if self.y_val is not None else None,
            ]

    def get_generators(
        self,
        batch_size: list or int,
        to_fit: list or bool,
    ) -> tuple:
        """Returns generators for the train, test and validation dataset.

        Args:
            batch_size (list or int): The batch size for each generator.
            to_fit (listorbool): Whether label information should be returned (if present).
            shuffle (list or bool): Whether data should be shuffled before returning.

        Returns:
            tuple: Tuple of generators for train, test and validation
        """
        if not isinstance(batch_size, list):
            batch_size = [batch_size] * 3

        if not isinstance(to_fit, list):
            to_fit = [to_fit] * 3

        train_generator = SimpleDataGenerator(
            self.X_train,
            self.y_train,
            batch_size[0],
            to_fit[0],
        )
        test_generator = SimpleDataGenerator(
            self.X_test,
            self.y_test,
            batch_size[1],
            to_fit[1],
        )
        val_generator = SimpleDataGenerator(
            self.X_val,
            self.y_val,
            batch_size[2],
            to_fit[2],
        )

        return train_generator, test_generator, val_generator

    def _load_data(self) -> None:
        """Loads data due to config and then sets the data of the DataContainer due to indices. Generates new split if no indices were provided.

        Raises:
            FileNotFoundError: Couldn't find the file for the first data path.
            FileNotFoundError: Couldn't find the file for the second data path, i.e., LDP.
            AssertionError: Loaded both datasets but they differ in length.
        """

        self._util.log(
            f"{self.__class__} loads data from: {self._data_path}", required_level=1
        )

        if not os.path.isfile(self._data_path):
            raise FileNotFoundError(
                f"{self.__class__} tried to load data from {self._data_path} but file is missing."
            )
        X, y = DataContainer.load_data(self._data_path)

        if self._eps_data_path:
            self._util.log(
                f"{self.__class__} loads data from: {self._eps_data_path}",
                required_level=1,
            )
            if not os.path.isfile(self._eps_data_path):
                raise FileNotFoundError(
                    f"{self.__class__} tried to load perturbed data from {self._eps_data_path} but file is missing."
                )

            X_perturbed, y_perturbed = DataContainer.load_data(self._eps_data_path)

            if not len(X_perturbed) == len(X):
                raise AssertionError(
                    f"{self.__class__} loaded original data from {self._data_path} and perturbed data from {self._eps_data_path} but they don't have equal length: {len(X)} vs. {len(X_perturbed)}"
                )

        if (not self._data_indices) or self._data_conf["force_new_data_indices"]:
            if self._data_conf["stratify_split"]:
                self._determine_stratified_train_test_val_split(y)
            else:
                self._determine_train_test_val_split(len(X))

        self._check_indices_validity(len(X))

        X_train = (
            X_perturbed[self._data_indices["train_indices"]]
            if self._data_conf["train"]
            else X[self._data_indices["train_indices"]]
        )

        X_test = (
            X_perturbed[self._data_indices["test_indices"]]
            if self._data_conf["test"]
            else X[self._data_indices["test_indices"]]
        )

        X_val = (
            X_perturbed[self._data_indices["val_indices"]]
            if self._data_conf["val"]
            else X[self._data_indices["val_indices"]]
        )
        # Not always label information present
        if len(y.shape) > 0:
            y_train = (
                y_perturbed[self._data_indices["train_indices"]]
                if self._data_conf["train"]
                else y[self._data_indices["train_indices"]]
            )

            y_test = (
                y_perturbed[self._data_indices["test_indices"]]
                if self._data_conf["test"]
                else y[self._data_indices["test_indices"]]
            )

            y_val = (
                y_perturbed[self._data_indices["val_indices"]]
                if self._data_conf["val"]
                else y[self._data_indices["val_indices"]]
            )
        else:
            y_train, y_test, y_val = None, None, None

        self._set_data(X_train, X_test, X_val, y_train, y_test, y_val)

    def _determine_upper_limit(self, key: str, length: int) -> int:

        dict_val = self._data_conf[key]

        if isinstance(dict_val, int):
            upper_limit = dict_val
        elif isinstance(dict_val, float):
            upper_limit = int(dict_val * length)
        else:
            upper_limit = int(0.1 * length)

        if batch_size := self._data_conf["align_with_batch_size"]:
            upper_limit = (upper_limit // batch_size) * batch_size

        return upper_limit

    def _determine_train_test_val_split(self, length: int) -> None:
        """Generates indices to divide dataset into train, test and validation sets due to config. Not stratified!

        Args:
            length (int): Length of the dataset to generate random indices for.
        """
        idx = np.arange(length)

        np.random.default_rng().shuffle(idx)

        # train_upper = int(length * (self._data_conf["train_size"] or 0.1))
        train_upper = self._determine_upper_limit("train_size", length)
        train_idx = idx[:train_upper]

        # val_upper = train_upper + int(length * (self._data_conf["val_size"] or 0.1))
        val_upper = self._determine_upper_limit("val_size", length)
        val_idx = idx[train_upper : train_upper + val_upper]

        test_idx = idx[train_upper + val_upper :]

        self._data_indices = {
            "train_indices": train_idx,
            "test_indices": test_idx,
            "val_indices": val_idx,
        }

    def _determine_stratified_train_test_val_split(self, labels: list) -> None:
        num_samples = len(labels)
        data_idx = np.arange(num_samples)

        num_train_samples = self._determine_upper_limit("train_size", num_samples)
        num_val_samples = self._determine_upper_limit("val_size", num_samples)

        first_sss = StratifiedShuffleSplit(n_splits=1, train_size=num_train_samples)
        split_train_idx, split_remain_idx = next(
            first_sss.split(np.zeros(num_samples), labels)
        )

        second_sss = StratifiedShuffleSplit(n_splits=1, train_size=num_val_samples)
        split_val_idx, split_test_idx = next(
            second_sss.split(np.zeros(len(split_remain_idx)), labels[split_remain_idx])
        )

        self._data_indices = {
            "train_indices": data_idx[split_train_idx],
            "test_indices": data_idx[split_remain_idx][split_test_idx],
            "val_indices": data_idx[split_remain_idx][split_val_idx],
        }

    def _check_indices_validity(self, length: int) -> None:

        if not (len_train_ind := len(self._data_indices["train_indices"])) == (
            len_train_conf := self._determine_upper_limit("train_size", length)
        ):
            raise RuntimeError(
                f"Length of provided train indices does not match config train size ({len_train_ind} vs {len_train_conf}). Either delete data_indices.npz file or set `force_new_data_indices` in config."
            )

        if not (len_val_ind := len(self._data_indices["val_indices"])) == (
            len_val_conf := self._determine_upper_limit("val_size", length)
        ):
            raise RuntimeError(
                f"Length of provided validation indices does not match config validation size ({len_val_ind} vs {len_val_conf}). Either delete data_indices.npz file or set `force_new_data_indices` in config."
            )

    def get_data_shape(self) -> tuple:
        """Returns the shape of the training data

        Returns:
            tuple: shape of the training data, test data and validation data
        """
        return self.X_train.shape, self.X_test.shape, self.X_val.shape

    def get_unique_labels(self) -> list or None:
        """Determines the unique labels in the data. If no label information is present, returns None.
        Returns:
            list or None: the unique labels or None.
        """
        all_ys = np.array([])

        if (ytr := self.y_train) is not None:
            # if labels are already one hot encoded, reverse
            if len(ytr.shape) > 1:
                ytr = np.argmax(ytr, axis=1)
            all_ys = np.concatenate([all_ys, ytr])

        if (yte := self.y_test) is not None:
            # if labels are already one hot encoded, reverse
            if len(yte.shape) > 1:
                yte = np.argmax(yte, axis=1)
            all_ys = np.concatenate([all_ys, yte])

        if (yval := self.y_val) is not None:
            # if labels are already one hot encoded, reverse
            if len(yval.shape) > 1:
                yval = np.argmax(yval, axis=1)
            all_ys = np.concatenate([all_ys, yval])

        unique_labels = np.unique(all_ys)
        return unique_labels if len(unique_labels) > 0 else None

    def get_num_unique_labels(self) -> int or None:
        """calculates how many unique labels are in the data. If no label information is present, returns None.
        Returns:
            int or None: the number of unique labels or None.
        """
        unique_labels = self.get_unique_labels()
        return len(unique_labels) if unique_labels is not None else None

    def get_figure_dir(self) -> str:
        """Creates if necessary and returns path to figure dir next to data files.

        Returns:
            str: Path to the directory.
        """

        figure_dir = os.path.join(os.path.dirname(self._data_path), "figures")

        if not os.path.isdir(figure_dir):
            os.makedirs(figure_dir)

        return figure_dir

    def plot_overall_label_histogram(self) -> None:
        """Checks if label information is present and plots overall corresponding distribution."""
        figure_dir = self.get_figure_dir()

        ys = []

        if (ytr := self.y_train) is not None:
            ys.append(ytr)

        if (yte := self.y_test) is not None:
            ys.append(yte)

        if (yval := self.y_val) is not None:
            ys.append(yval)

        if len(ys) == 0:
            return

        ys = np.concatenate(ys)

        # if labels are already one hot encoded, reverse
        if len(ys.shape) > 1:
            ys = np.argmax(ys, axis=1)

        figure_helper.plot_data_as_histogram(
            ys,
            xlabel="labels",
            savepath=os.path.join(figure_dir, "label_distribution.pdf"),
        )

    def plot_label_histogram(self, figure_dir: str) -> None:
        """Checks if label information is present and plots the corresponding distribution for train test and validation.

        Args.
            figure_dir (str): path to save figures to.
        """

        if (ytr := self.y_train) is not None:
            # if labels are already one hot encoded, reverse
            if len(ytr.shape) > 1:
                ytr = np.argmax(ytr, axis=1)

            figure_helper.plot_data_as_histogram(
                ytr,
                xlabel="train labels",
                savepath=os.path.join(figure_dir, "train_label_distribution.pdf"),
            )

        if (yte := self.y_test) is not None:
            # if labels are already one hot encoded, reverse
            if len(yte.shape) > 1:
                yte = np.argmax(yte, axis=1)

            figure_helper.plot_data_as_histogram(
                yte,
                xlabel="test labels",
                savepath=os.path.join(figure_dir, "test_label_distribution.pdf"),
            )

        if (yval := self.y_val) is not None:
            # if labels are already one hot encoded, reverse
            if len(yval.shape) > 1:
                yval = np.argmax(yval, axis=1)

            figure_helper.plot_data_as_histogram(
                yval,
                xlabel="validation labels",
                savepath=os.path.join(figure_dir, "val_label_distribution.pdf"),
            )

    @staticmethod
    def save_data(data_path: str, X: list, y: list) -> None:
        """Saves a complete dataset in compressed manner.

        Args:
            data_path (str): path to save data to.
            X (list): Data features.
            y (list): Data labels.
        """
        if (len_x := len(X)) < 1:
            raise ValueError(f"Passed empty dataset (len: {len_x}) to save_data!")

        data_dir = os.path.dirname(data_path)
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        np.savez_compressed(data_path, X=X, y=y)

    @staticmethod
    def load_data(data_path) -> Tuple[np.array, np.array]:
        """Loads a complete dataset.

        Args:
            data_path ([type]): Path to load data from.

        Returns:
            Tuple[np.array, np.array]: Features and labels.
        """

        with np.load(data_path, allow_pickle=True) as data:
            X, y = data["X"], data["y"]

        if (len_x := len(X)) < 1:
            raise ValueError(
                f"Tried to load empty dataset (len: {len_x}) in load_data!"
            )

        return X, y

    """
    Methods to be defined in the subclasses for the concrete data types
    """

    @abstractmethod
    def preprocess_data():
        """Called by ModelContainer to get train-ready data."""
        pass

    @staticmethod
    @abstractmethod
    def perturb():
        """Static method to create a missing LDP dataset from a datapath."""
        pass

    @staticmethod
    @abstractmethod
    def get_data():
        """Static method to create a missing dataset."""
        pass


class BaseImageClass(DataContainer, ABC):
    """Image specific base class. Can be used to distinguish in model evaluations"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class BaseTimeSeriesClass(DataContainer, ABC):
    """Time Series specific base class. Can be used to distinguish in model evaluations"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class PictureContainer(BaseImageClass):
    """Picture specific container class"""

    def __init__(
        self,
        util_ref: object,
        data_dir: str,
        dataset: str,
        data_conf: Dict,
        perturbation_conf: Dict = None,
        data_indices: Dict = None,
    ) -> None:
        """Initialize for a PictureContainer. If the provided data_path is missing, it downloads and creates the missing dataset.

        Args:
            util_ref (object): Reference to BaseHelper, necessary for logging.
            data_dir (str): Path to the overall data directory.
            dataset (str): Name of the wanted dataset.
            data_conf (Dict): Dictionary with config for dataset creation or loading.
            perturbation_conf (Dict, optional): Dictionary with config for dataset perturbation. Defaults to None.
            data_indices (Dict, optional): Dict with indices to recreate a specific data split. Defaults to None.
        """

        data_path = os.path.join(data_dir, dataset, f"{dataset}.npz")
        if not os.path.isfile(data_path):
            util_ref.log(
                f"{self.__class__} creating and saving {dataset} to {data_path}", 1
            )
            PictureContainer.get_data(data_path, dataset)

        if noise := perturbation_conf["data_ldp_noise"]:
            eps_data_path = os.path.join(
                data_dir,
                dataset,
                f"{dataset}_noise_{noise}.npz",
            )

            if not os.path.isfile(eps_data_path):
                raise RuntimeError(
                    f"Asked for VAE-LDP data with noise `{noise}` but not present at `{eps_data_path}`. Train a separate VAE and perturb dataset first."
                )

        elif eps := perturbation_conf["epsilon"]:
            eps_data_path = os.path.join(
                data_dir,
                dataset,
                f"{dataset}_eps_{eps}_m_{perturbation_conf['m']}_b_{perturbation_conf['b'] if 'b' in perturbation_conf.keys() else 1 }.npz",
            )

            if not os.path.isfile(eps_data_path):
                util_ref.log(
                    f"{self.__class__} perturbing data from {data_path} to {eps_data_path}",
                    1,
                )

                PictureContainer.perturb(
                    data_path,
                    eps_data_path,
                    perturbation_conf["epsilon"],
                    perturbation_conf["m"],
                    perturbation_conf["b"],
                )

        else:
            eps_data_path = None

        super().__init__(util_ref, data_path, data_conf, eps_data_path, data_indices)

    def preprocess_data(self, model_type: str, dim_y: int = 100) -> None:
        """Prepare picture data for specific model.
        here, the following steps are performed.
            1. Transform pixel values. [-1, 1] for GAN and [0, 1] for VAE.
            2. Add dummy dimension for alpha channel in RGB pictures (GAN)
            3. Create One-hot encoding for labels.

        Args:
            model_type (str): Modelclass.
            dim_y (int, optional): Dimension for the encoded labels. Defaults to 100.

        Raises:
            ValueError: Unknown model_type.
        """

        X_train, X_test, X_val, y_train, y_test, y_val = self.unravel()

        # TODO Handle cifar10 data (color channel dim = 3)

        if model_type == "GAN":

            if X_train.ndim == 3:
                # Grayscale data ([0, 255]) -> transform ([-1, 1])
                X_train = util.transform_images_minus_one_one_range(X_train)
                X_test = util.transform_images_minus_one_one_range(X_test)
                X_val = util.transform_images_minus_one_one_range(X_val)

                # Add dim (colour channel)
                X_train = X_train[:, :, :, None]
                X_test = X_test[:, :, :, None]
                X_val = X_val[:, :, :, None]

            # One-hot encode labels
            # ! FIXME Remove 100/dim_y dim restraint once possible
            y_train = to_categorical(y_train, dim_y)
            y_test = to_categorical(y_test, dim_y)
            y_val = to_categorical(y_val, dim_y)
            # y_train = to_categorical(y_train)
            # y_test = to_categorical(y_test)
            # y_val = to_categorical(y_val)

        elif model_type == "VAE":

            # Transform
            X_train = util.transform_images_zero_one_range(X_train)
            X_test = util.transform_images_zero_one_range(X_test)
            X_val = util.transform_images_zero_one_range(X_val)

            # Is now done right before model training
            # Flatten
            # X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
            # X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
            # X_val = X_val.reshape((len(X_val), np.prod(X_val.shape[1:])))

            # One-hot encode labels
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
            y_val = to_categorical(y_val)

        else:
            raise ValueError(f"Unknown model type for data preprocessing: {model_type}")

        self._set_data(X_train, X_test, X_val, y_train, y_test, y_val)

    def _set_data(
        self,
        X_train: np.array,
        X_test: np.array,
        X_val: np.array,
        y_train: np.array,
        y_test: np.array,
        y_val: np.array,
    ) -> None:
        """Add Picture specific checks before setting data.

        Args:
            X_train (np.array): Records supposed for training.
            X_test (np.array): Records supposed for testing.
            X_val (np.array): Records supposed for validation.
            y_train (np.array): Corresponding labels for training data.
            y_test (np.array): Corresponding labels for test data.
            y_val (np.array): Corresponding labels for validation data.

        Raises:
            ValueError: If data shape is not quadratic.
        """

        self._util.log(f"{self.__class__} sets its own data.", required_level=2)

        if not util.has_quadratic_shape(X_train):
            raise ValueError(
                f"{self.__class__} expects quadratic picture shape, but is supposed to set x_train with `{X_train.shape}` "
            )

        if not util.has_quadratic_shape(X_test):
            raise ValueError(
                f"{self.__class__} expects quadratic picture shape, but is supposed to set x_test with `{X_test.shape}` "
            )

        if not util.has_quadratic_shape(X_val):
            raise ValueError(
                f"{self.__class__} expects quadratic picture shape, but is supposed to set x_val with `{X_val.shape}` "
            )

        super()._set_data(X_train, X_test, X_val, y_train, y_test, y_val)

    @staticmethod
    def perturb(
        data_path: str, eps_data_path: str, epsilon: float, m: int, b: int = 1
    ) -> None:
        """Perturb Pictures with pixelization
        Args:
            data_path (str): Path to load data from.
            eps_data_path (str): Path to save perturbed data to.
            epsilon (float): Epsilon value.
            m (int): Number of different pixels allowed, i.e., neighborhood.
            b (int): Grid cell length. Defaults to 1.
        """

        if not b:
            b = 1

        # anonymous methods for pixelization
        def get_pixelgroup(img, b):
            width, height = img.shape
            for w in range(0, width - b, b):
                for h in range(0, height - b, b):
                    group = img[w : w + b, h : h + b]
                    gw, gh = group.shape
                    yield group.reshape(gw * gh)

                # edge case: height % b != 0, thus we shift b back
                group = img[w : w + b, -b:]
                gw, gh = group.shape
                yield group.reshape(gw * gh)

            # edge case: width % b != 0, thus we shift b back
            for h in range(0, height - b, b):
                group = img[-b:, h : h + b]
                gw, gh = group.shape
                yield group.reshape(gw * gh)

            # edge case: width and height % b != 0, thus we shift b back
            group = img[-b:, -b:]
            gw, gh = group.shape
            yield group.reshape(gw * gh)

        def perturb(value, m, b, eps):
            mu = 0
            scale = (255 * m) / ((b * b) * eps)
            rand = np.random.default_rng().laplace(mu, scale)
            value = np.clip(0, 255, value + rand)  # truncate [0...255]
            return value

        X, y = DataContainer.load_data(data_path)

        X_perturbed = []
        for img in tqdm(X):
            X_perturbed.append(
                np.array(
                    list(
                        map(
                            np.round,
                            map(
                                lambda x: perturb(x, m, b, epsilon),
                                map(np.mean, get_pixelgroup(img, b)),
                            ),
                        )
                    ),
                    dtype=np.int,
                )
            )
        X_perturbed = np.array(X_perturbed)

        DataContainer.save_data(eps_data_path, X_perturbed, y)

    @staticmethod
    def get_data(data_path: str, dataset: str):
        """Obtain pictures using the keras data set loader and save them locally.

        Args:
            data_path (str): Path to save data to.
            dataset (str): Identifier for the data set. Either 'mnist', 'fashion_mnist' or 'cifar10'.
        """

        if dataset not in globals().keys():
            raise NotImplementedError(
                f"Picture Container called with unknown dataset `{dataset}`"
            )

        (X1, y1), (X2, y2) = globals()[dataset].load_data()

        # Join test and training set back together to enforce own split sizes
        X = np.concatenate([X1, X2])
        y = np.concatenate([y1, y2])

        # Remove unnecessary dimensions
        y = np.squeeze(y)

        DataContainer.save_data(data_path, X, y)


class DistributionContainer(BaseImageClass):
    def __init__(
        self,
        util_ref: object,
        data_dir: str,
        dataset: str,
        data_conf: Dict,
        perturbation_conf: Dict = None,
        data_indices: Dict = None,
    ) -> None:

        data_path = os.path.join(data_dir, dataset, f"{dataset}.npz")
        if not os.path.isfile(data_path):
            util_ref.log(
                f"{self.__class__} creating and saving {dataset} to {data_path}", 1
            )
            DistributionContainer.get_data(data_path, dataset)

        eps_data_path = None
        if perturbation_conf["epsilon"]:
            DistributionContainer.perturb(
                data_path, eps_data_path, perturbation_conf["epsilon"]
            )

        super().__init__(util_ref, data_path, data_conf, eps_data_path, data_indices)

    def preprocess_data(self, *args, **kwargs) -> None:
        """Preprocess data in the container. Currently, we do not need any preprocessing for Distribution data."""
        pass

    @staticmethod
    def perturb(data_path: str, eps_data_path: str, epsilon: float) -> None:
        """Perturbation for Distribution data is currently not supported."""

        raise NotImplementedError(
            f"DistributionContainer does not support Perturbation."
        )

    @staticmethod
    def get_data(data_path: str, dataset: str, num_samples: int = 100000):
        """[summary]

        Args:
            data_path (str): [description]
            dataset (str): Name of the dataset. May prepended with dimension and append with seed. e.g., 2_multivariate_normal_1234
            num_samples (int, optional): Number of records. Defaults to 100000.

        Raises:
            NotImplementedError: Raised, when the dataset is unknown
        """
        dataset = dataset.split("_")
        seed = None
        if dataset[-1].isnumeric():
            seed = int(dataset[-1])
            dataset = dataset[:-1]
        else:
            seed = np.random.default_rng().integers(99999999)

        dim = None
        if dataset[0].isnumeric():
            dim = int(dataset[0])
            dataset = dataset[1:]
        else:
            dim = 1

        dataset = "_".join(dataset)

        if dataset not in globals().keys():
            raise NotImplementedError(
                f"DistributionContainer called with unknown dataset `{dataset}` for get_data"
            )

        rng = np.random.default_rng(seed)
        mu = rng.uniform(-3, 3, dim)
        if dim > 1:
            A = rng.random((dim, dim))
            sigma = np.dot(A, A.transpose())
        else:
            sigma = rng.uniform(0, 3, dim)

        self._util.log(f"mu: {mu}\tsigma: {sigma}", 1)

        X = globals()[dataset](mu, sigma, num_samples)

        DataContainer.save_data(data_path, X, None)


# Path to the raw celebA pictures
CELEBA_RAW_PICTURE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/celebA/data/img_celeba_align",
)


class CelebAContainer(BaseImageClass):
    def __init__(
        self,
        util_ref: object,
        data_dir: str,
        dataset: str,
        data_conf: Dict,
        perturbation_conf: Dict = None,
        data_indices: Dict = None,
    ) -> None:
        """Initialize for a CelebAContainer. If the provided data_path is missing, it creates the missing dataset.

        Args:
            util_ref (object): Reference to BaseHelper, necessary for logging.
            data_dir (str): Path to the overall data directory.
            dataset (str): Name of the wanted dataset.
            data_conf (Dict): Dictionary with config for dataset creation or loading.
            perturbation_conf (Dict, optional): Dictionary with config for dataset perturbation. Defaults to None.
            data_indices (Dict, optional): Dict with indices to recreate a specific data split. Defaults to None.
        """

        data_path = os.path.join(data_dir, dataset, f"{dataset}.npz")
        if not os.path.isfile(data_path):
            util_ref.log(
                f"{self.__class__} creating and saving {dataset} to {data_path}", 1
            )
            CelebAContainer.get_data(data_path, dataset)

        if noise := perturbation_conf["data_ldp_noise"]:
            eps_data_path = os.path.join(
                data_dir,
                dataset,
                f"{dataset}_noise_{noise}.npz",
            )

            if not os.path.isfile(eps_data_path):
                raise RuntimeError(
                    f"Asked for VAE-LDP data with noise `{noise}` but not present at `{eps_data_path}`. Train a separate VAE and perturb dataset first."
                )

        elif eps := perturbation_conf["epsilon"]:
            eps_data_path = os.path.join(
                data_dir,
                dataset,
                f"{dataset}_eps_{eps}_m_{perturbation_conf['m']}_b_{perturbation_conf['b'] if perturbation_conf['b'] is not None else 1 }.npz",
            )

            if not os.path.isfile(eps_data_path):
                util_ref.log(
                    f"{self.__class__} perturbing data from {data_path} to {eps_data_path}",
                    1,
                )

                CelebAContainer.perturb(
                    data_path,
                    eps_data_path,
                    perturbation_conf["epsilon"],
                    perturbation_conf["m"],
                    perturbation_conf["b"],
                )

        else:
            eps_data_path = None

        super().__init__(util_ref, data_path, data_conf, eps_data_path, data_indices)

    def preprocess_data(self, model_type: str, dim_y: int = 100) -> None:
        """Prepare picture data for specific model.
        here, the following steps are performed.
            1. Transform pixel values. [0, 1] for VAE.
            2. Create One-hot encoding for labels.

        Args:
            model_type (str): Only for compability..
            dim_y (int, optional): Only for compability..

        """

        X_train, X_test, X_val, y_train, y_test, y_val = self.unravel()

        # scale image between 0 & 1
        X_train = util.transform_images_zero_one_range(X_train)
        X_test = util.transform_images_zero_one_range(X_test)
        X_val = util.transform_images_zero_one_range(X_val)

        num_classes = self.get_num_unique_labels()
        if y_train is not None:
            y_train = to_categorical(y_train, num_classes=num_classes)

        if y_test is not None:
            y_test = to_categorical(y_test, num_classes=num_classes)

        if y_val is not None:
            y_val = to_categorical(y_val, num_classes=num_classes)

        self._set_data(X_train, X_test, X_val, y_train, y_test, y_val)

    @staticmethod
    def perturb(
        data_path: str, eps_data_path: str, epsilon: float, m: int, b: int = 1
    ) -> None:

        """Perturb Colorimages (i.e., 3 chanel) with pixelization
        Args:
            data_path (str): Path to load data from.
            eps_data_path (str): Path to save perturbed data to.
            epsilon (float): Epsilon value.
            m (int): Number of different pixels allowed, i.e., neighborhood.
            b (int): Grid cell length. Defaults to 1.
        """

        if not b:
            b = 1

        def get_pixelgroup(img, b):
            width, height = img.shape[:2]
            for w in range(0, width - b, b):
                for h in range(0, height - b, b):
                    group = img[w : w + b, h : h + b]
                    yield group

                # edge case: height % b != 0, thus we shift b back
                group = img[w : w + b, -b:]
                yield group

            # edge case: width % b != 0, thus we shift b back
            for h in range(0, height - b, b):
                group = img[-b:, h : h + b]
                yield group

            # edge case: width and height % b != 0, thus we shift b back
            group = img[-b:, -b:]
            yield group

        def perturb(value, m, b, eps):
            mu = 0
            scale = (255 * m) / ((b * b) * eps)
            rand = np.random.default_rng().laplace(mu, scale, 3)
            value = np.clip(0, 255, value + rand)  # truncate [0...255]
            return value

        # allow pickle of anonymous function
        global get_perturbed_img

        def get_perturbed_img(img):
            perturbed_img = np.array(
                list(
                    map(
                        lambda x: perturb(x, m, b, epsilon),
                        map(
                            lambda x: list(np.mean(x, axis=(0, 1))),
                            get_pixelgroup(img, b),
                        ),
                    )
                ),
                dtype=np.uint8,
            )

            perturbed_shape = perturbed_img.shape
            size = int(np.ceil(np.sqrt(perturbed_shape[0])))

            return perturbed_img.reshape(size, size, perturbed_shape[-1])

        X, y = DataContainer.load_data(data_path)

        num_processes = len(os.sched_getaffinity(0))

        start = datetime.now()
        X_perturbed = process_map(
            get_perturbed_img, X, max_workers=num_processes, chunksize=1
        )
        stop = datetime.now()
        print(f"\nPerturbation took {stop - start}", flush=True)

        X_perturbed = np.array(X_perturbed)

        DataContainer.save_data(eps_data_path, X_perturbed, y)

    @staticmethod
    def get_data(data_path: str, dataset: str):

        dataset_split = dataset.split("-")
        width, height = int(dataset_split[-2]), int(dataset_split[-1])

        # needed for process map to pickle function, cannot pickle local function
        global convert_single_image

        def convert_single_image(img_name: str) -> np.ndarray:
            img_path = os.path.join(CELEBA_RAW_PICTURE_PATH, img_name)
            pil_img = Image.open(img_path)
            pil_cropped_img = pil_img.resize((width, height))
            return np.array(pil_cropped_img)

        start = datetime.now()

        # ? more efficient by changing to iglob, i.e., iterator
        # ? main problem: process map needs `total` param
        all_img_names = glob.glob(os.path.join(CELEBA_RAW_PICTURE_PATH, "*.png"))

        start_map = datetime.now()
        num_processes = len(os.sched_getaffinity(0))
        res = process_map(
            convert_single_image,
            all_img_names,
            max_workers=num_processes,
            chunksize=1,
        )
        stop_map = datetime.now()
        print(f"\nMap took {stop_map - start_map}", flush=True)

        stop = datetime.now()

        print(f"\nOverall calculation took {stop - start}", flush=True)

        X = np.array(res)
        DataContainer.save_data(data_path, X, None)


# Path to the raw lfw pictures: http://vis-www.cs.umass.edu/lfw/
LFW_RAW_PICTURE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/lfw/lfw-deepfunneled",
)


class LFWContainer(BaseImageClass):
    def __init__(
        self,
        util_ref: object,
        data_dir: str,
        dataset: str,
        data_conf: Dict,
        perturbation_conf: Dict = None,
        data_indices: Dict = None,
    ) -> None:
        """Initialize for a LFWContainer. If the provided data_path is missing, it creates the missing dataset.

        Args:
            util_ref (object): Reference to BaseHelper, necessary for logging.
            data_dir (str): Path to the overall data directory.
            dataset (str): Name of the wanted dataset.
            data_conf (Dict): Dictionary with config for dataset creation or loading.
            perturbation_conf (Dict, optional): Dictionary with config for dataset perturbation. Defaults to None.
            data_indices (Dict, optional): Dict with indices to recreate a specific data split. Defaults to None.
        """

        data_path = os.path.join(data_dir, dataset, f"{dataset}.npz")
        if not os.path.isfile(data_path):
            util_ref.log(
                f"{self.__class__} creating and saving {dataset} to {data_path}", 1
            )
            LFWContainer.get_data(data_path, dataset)

        if noise := perturbation_conf["data_ldp_noise"]:
            eps_data_path = os.path.join(
                data_dir,
                dataset,
                f"{dataset}_noise_{noise}.npz",
            )

            if not os.path.isfile(eps_data_path):
                raise RuntimeError(
                    f"Asked for VAE-LDP data with noise `{noise}` but not present at `{eps_data_path}`. Train a separate VAE and perturb dataset first."
                )

        elif eps := perturbation_conf["epsilon"]:
            eps_data_path = os.path.join(
                data_dir,
                dataset,
                f"{dataset}_eps_{eps}_m_{perturbation_conf['m']}_b_{perturbation_conf['b'] if perturbation_conf['b'] is not None else 1 }.npz",
            )

            if not os.path.isfile(eps_data_path):
                util_ref.log(
                    f"{self.__class__} perturbing data from {data_path} to {eps_data_path}",
                    1,
                )

                LFWContainer.perturb(
                    data_path,
                    eps_data_path,
                    perturbation_conf["epsilon"],
                    perturbation_conf["m"],
                    perturbation_conf["b"],
                )

        else:
            eps_data_path = None

        super().__init__(util_ref, data_path, data_conf, eps_data_path, data_indices)

    def preprocess_data(self, model_type: str, dim_y: int = 100) -> None:
        """Prepare picture data for specific model.
        here, the following steps are performed.
            1. Transform pixel values [0, 1] for VAE.
            2. Create One-hot encoding for labels.

        Args:
            model_type (str): Only for compability.
            dim_y (int, optional): Only for compability.
        """

        X_train, X_test, X_val, y_train, y_test, y_val = self.unravel()

        # scale image between 0 & 1
        X_train = util.transform_images_zero_one_range(X_train)
        X_test = util.transform_images_zero_one_range(X_test)
        X_val = util.transform_images_zero_one_range(X_val)

        num_classes = self.get_num_unique_labels()
        if y_train is not None:
            y_train = to_categorical(y_train, num_classes=num_classes)

        if y_test is not None:
            y_test = to_categorical(y_test, num_classes=num_classes)

        if y_val is not None:
            y_val = to_categorical(y_val, num_classes=num_classes)

        self._set_data(X_train, X_test, X_val, y_train, y_test, y_val)

    @staticmethod
    def perturb(
        data_path: str, eps_data_path: str, epsilon: float, m: int, b: int = 1
    ) -> None:

        """Perturb Colorimages (i.e., 3 chanel) with pixelization
        Args:
            data_path (str): Path to load data from.
            eps_data_path (str): Path to save perturbed data to.
            epsilon (float): Epsilon value.
            m (int): Number of different pixels allowed, i.e., neighborhood.
            b (int): Grid cell length. Defaults to 1.
        """

        if not b:
            b = 1

        def get_pixelgroup(img, b):
            width, height = img.shape[:2]
            for w in range(0, width - b, b):
                for h in range(0, height - b, b):
                    group = img[w : w + b, h : h + b]
                    yield group

                # edge case: height % b != 0, thus we shift b back
                group = img[w : w + b, -b:]
                yield group

            # edge case: width % b != 0, thus we shift b back
            for h in range(0, height - b, b):
                group = img[-b:, h : h + b]
                yield group

            # edge case: width and height % b != 0, thus we shift b back
            group = img[-b:, -b:]
            yield group

        def perturb(value, m, b, eps):
            mu = 0
            scale = (255 * m) / ((b * b) * eps)
            rand = np.random.default_rng().laplace(mu, scale, 3)
            value = np.clip(0, 255, value + rand)  # truncate [0...255]
            return value

        # allow pickle of anonymous function
        global get_perturbed_img

        def get_perturbed_img(img):
            perturbed_img = np.array(
                list(
                    map(
                        lambda x: perturb(x, m, b, epsilon),
                        map(
                            lambda x: list(np.mean(x, axis=(0, 1))),
                            get_pixelgroup(img, b),
                        ),
                    )
                ),
                dtype=np.uint8,
            )

            perturbed_shape = perturbed_img.shape
            size = int(np.ceil(np.sqrt(perturbed_shape[0])))

            return perturbed_img.reshape(size, size, perturbed_shape[-1])

        X, y = DataContainer.load_data(data_path)

        num_processes = len(os.sched_getaffinity(0))

        start = datetime.now()
        X_perturbed = process_map(
            get_perturbed_img, X, max_workers=num_processes, chunksize=1
        )
        stop = datetime.now()
        print(f"\nPerturbation took {stop - start}", flush=True)

        X_perturbed = np.array(X_perturbed)

        DataContainer.save_data(eps_data_path, X_perturbed, y)

    @staticmethod
    def get_data(data_path: str, dataset: str):
        # assume name to be LFW-width-height or LFW-width-height-classes
        dataset_split = dataset.split("-")
        width, height, classes = None, None, None

        if len(dataset_split) > 3:
            width, height, classes = (
                int(dataset_split[-3]),
                int(dataset_split[-2]),
                int(dataset_split[-1]),
            )
        else:
            width, height = int(dataset_split[-2]), int(dataset_split[-1])

        # needed for process map to pickle function, cannot pickle local function
        global convert_single_image

        def convert_single_image(img_name: str) -> list:
            pil_img = Image.open(img_name)
            pil_cropped_img = pil_img.resize((width, height))
            return [np.array(pil_cropped_img)[np.newaxis, :], img_name.split("/")[-2]]

        start = datetime.now()

        # ? more efficient by changing to iglob, i.e., iterator
        # ? main problem: process map needs `total` param
        all_img_names = glob.glob(os.path.join(LFW_RAW_PICTURE_PATH, "*", "*.jpg"))
        num_processes = len(os.sched_getaffinity(0))

        res = process_map(
            convert_single_image,
            all_img_names,
            max_workers=num_processes,
            chunksize=1,
        )

        res = np.array(res)

        # replace label with
        unique_labels = np.unique(res[:, 1])
        for idx, label in tenumerate(unique_labels):
            res[:, 1][res[:, 1] == label] = idx

        X, y = np.concatenate(res[:, 0]), res[:, 1]

        if classes:
            # shift classes up
            shift_value = max(y) + 1
            y += shift_value
            unique_labels, count = np.unique(y, return_counts=True)
            idx = np.argsort(count)[-1 * classes :]
            # re-label
            for i, label in tenumerate(unique_labels[idx]):
                y[res[:, 1] == label] = i

            mask = y < shift_value
            X, y = X[mask], y[mask]

        stop = datetime.now()
        print(f"\nOverall calculation took {stop - start}", flush=True)

        DataContainer.save_data(data_path, X, y)


# Path to the raw celebA pictures
CELEBA_RAW_CONDITIONAL_PICTURE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/celebraw/img_align_celeba",
)

# Path to the celebA labels
CELEBA_RAW_CONDITIONAL_LABELS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/celebraw/list_attr_celeba_no_ws.txt",
)


class CelebAConditionalContainer(BaseImageClass):
    def __init__(
        self,
        util_ref: object,
        data_dir: str,
        dataset: str,
        data_conf: Dict,
        perturbation_conf: Dict = None,
        data_indices: Dict = None,
    ) -> None:
        """Initialize for a CelebAConditionalContainer. If the provided data_path is missing, it creates the missing dataset.

        Args:
            util_ref (object): Reference to BaseHelper, necessary for logging.
            data_dir (str): Path to the overall data directory.
            dataset (str): Name of the wanted dataset.
            data_conf (Dict): Dictionary with config for dataset creation or loading.
            perturbation_conf (Dict, optional): Dictionary with config for dataset perturbation. Defaults to None.
            data_indices (Dict, optional): Dict with indices to recreate a specific data split. Defaults to None.
        """

        data_path = os.path.join(data_dir, dataset, f"{dataset}.npz")
        if not os.path.isfile(data_path):
            util_ref.log(
                f"{self.__class__} creating and saving {dataset} to {data_path}", 1
            )
            CelebAConditionalContainer.get_data(data_path, dataset)

        if noise := perturbation_conf["data_ldp_noise"]:
            eps_data_path = os.path.join(
                data_dir,
                dataset,
                f"{dataset}_noise_{noise}.npz",
            )

            if not os.path.isfile(eps_data_path):
                raise RuntimeError(
                    f"Asked for VAE-LDP data with noise `{noise}` but not present at `{eps_data_path}`. Train a separate VAE and perturb dataset first."
                )

        elif eps := perturbation_conf["epsilon"]:
            eps_data_path = os.path.join(
                data_dir,
                dataset,
                f"{dataset}_eps_{eps}_m_{perturbation_conf['m']}_b_{perturbation_conf['b'] if perturbation_conf['b'] is not None else 1 }.npz",
            )

            if not os.path.isfile(eps_data_path):
                util_ref.log(
                    f"{self.__class__} perturbing data from {data_path} to {eps_data_path}",
                    1,
                )

                CelebAConditionalContainer.perturb(
                    data_path,
                    eps_data_path,
                    perturbation_conf["epsilon"],
                    perturbation_conf["m"],
                    perturbation_conf["b"],
                )

        else:
            eps_data_path = None

        super().__init__(util_ref, data_path, data_conf, eps_data_path, data_indices)

    def preprocess_data(self, model_type: str, dim_y: int = 100) -> None:
        """Prepare picture data for specific model.
        here, the following steps are performed.
            1. Transform pixel values. [0, 1] for VAE.
            2. Create One-hot encoding for labels.

        Args:
            model_type (str): Only for compability..
            dim_y (int, optional): Only for compability..

        """

        X_train, X_test, X_val, y_train, y_test, y_val = self.unravel()

        # scale image between 0 & 1
        X_train = util.transform_images_zero_one_range(X_train)
        X_test = util.transform_images_zero_one_range(X_test)
        X_val = util.transform_images_zero_one_range(X_val)

        self._set_data(X_train, X_test, X_val, y_train, y_test, y_val)

    @staticmethod
    def perturb(
        data_path: str, eps_data_path: str, epsilon: float, m: int, b: int = 1
    ) -> None:

        """Perturb Colorimages (i.e., 3 chanel) with pixelization
        Args:
            data_path (str): Path to load data from.
            eps_data_path (str): Path to save perturbed data to.
            epsilon (float): Epsilon value.
            m (int): Number of different pixels allowed, i.e., neighborhood.
            b (int): Grid cell length. Defaults to 1.
        """

        if not b:
            b = 1

        def get_pixelgroup(img, b):
            width, height = img.shape[:2]
            for w in range(0, width - b, b):
                for h in range(0, height - b, b):
                    group = img[w : w + b, h : h + b]
                    yield group

                # edge case: height % b != 0, thus we shift b back
                group = img[w : w + b, -b:]
                yield group

            # edge case: width % b != 0, thus we shift b back
            for h in range(0, height - b, b):
                group = img[-b:, h : h + b]
                yield group

            # edge case: width and height % b != 0, thus we shift b back
            group = img[-b:, -b:]
            yield group

        def perturb(value, m, b, eps):
            mu = 0
            scale = (255 * m) / ((b * b) * eps)
            rand = np.random.default_rng().laplace(mu, scale, 3)
            value = np.clip(0, 255, value + rand)  # truncate [0...255]
            return value

        # allow pickle of anonymous function
        global get_perturbed_img

        def get_perturbed_img(img):
            perturbed_img = np.array(
                list(
                    map(
                        lambda x: perturb(x, m, b, epsilon),
                        map(
                            lambda x: list(np.mean(x, axis=(0, 1))),
                            get_pixelgroup(img, b),
                        ),
                    )
                ),
                dtype=np.uint8,
            )

            perturbed_shape = perturbed_img.shape
            size = int(np.ceil(np.sqrt(perturbed_shape[0])))

            return perturbed_img.reshape(size, size, perturbed_shape[-1])

        X, y = DataContainer.load_data(data_path)

        num_processes = len(os.sched_getaffinity(0))

        start = datetime.now()
        X_perturbed = process_map(
            get_perturbed_img, X, max_workers=num_processes, chunksize=1
        )
        stop = datetime.now()
        print(f"\nPerturbation took {stop - start}", flush=True)

        X_perturbed = np.array(X_perturbed)

        DataContainer.save_data(eps_data_path, X_perturbed, y)

    @staticmethod
    def get_data(data_path: str, dataset: str):
        dataset_split = dataset.split("-")

        width, height = int(dataset_split[-2]), int(dataset_split[-1])

        # needed for process map to pickle function, cannot pickle local function
        global convert_single_image

        def convert_single_image(img_name: str) -> np.ndarray:
            img_path = os.path.join(CELEBA_RAW_CONDITIONAL_PICTURE_PATH, img_name)
            pil_img = Image.open(img_path)
            pil_cropped_img = pil_img.resize((width, height))
            return np.array(pil_cropped_img)

        start = datetime.now()

        # ? more efficient by changing to iglob, i.e., iterator
        # ? main problem: process map needs `total` param
        all_img_names = glob.glob(
            os.path.join(CELEBA_RAW_CONDITIONAL_PICTURE_PATH, "*.jpg")
        )

        start_map = datetime.now()
        num_processes = len(os.sched_getaffinity(0))
        res = process_map(
            convert_single_image,
            all_img_names,
            max_workers=num_processes,
            chunksize=1,
        )
        stop_map = datetime.now()
        print(f"\nMap took {stop_map - start_map}", flush=True)

        stop = datetime.now()

        print(f"\nOverall calculation took {stop - start}", flush=True)

        y = np.loadtxt(
            CELEBA_RAW_CONDITIONAL_LABELS_PATH, delimiter=",", skiprows=1, usecols=(3)
        )

        # Replace -1 by 0
        for i, val in enumerate(y):
            if val == -1:
                y[i] = 0

        X = np.array(res)

        DataContainer.save_data(data_path, X, y)


MOTIONSENSE_RAW_SENSOR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/A_DeviceMotion_data/",
)


class MotionSenseConditionalContainer(BaseTimeSeriesClass):
    def __init__(
        self,
        util_ref: object,
        data_dir: str,
        dataset: str,
        data_conf: Dict,
        perturbation_conf: Dict = None,
        data_indices: Dict = None,
    ) -> None:
        """Initialize for a MotionSenseConditionalContainer. If the provided data_path is missing, it creates the missing dataset.
        Args:
            util_ref (object): Reference to BaseHelper, necessary for logging.
            data_dir (str): Path to the overall data directory.
            dataset (str): Name of the wanted dataset.
            data_conf (Dict): Dictionary with config for dataset creation or loading.
            perturbation_conf (Dict, optional): Dictionary with config for dataset perturbation. Defaults to None.
            data_indices (Dict, optional): Dict with indices to recreate a specific data split. Defaults to None.
        """

        data_path = os.path.join(data_dir, dataset, f"{dataset}.npz")
        self.data_path = data_path

        if not os.path.isfile(data_path):
            util_ref.log(
                f"{self.__class__} creating and saving {dataset} to {data_path}", 1
            )
            MotionSenseConditionalContainer.get_data(data_path, dataset)

        if noise := perturbation_conf["data_ldp_noise"]:
            eps_data_path = os.path.join(
                data_dir,
                dataset,
                f"{dataset}_noise_{noise}.npz",
            )

            if not os.path.isfile(eps_data_path):
                raise RuntimeError(
                    f"Asked for VAE-LDP data with noise `{noise}` but not present at `{eps_data_path}`. Train a separate VAE and perturb dataset first."
                )

        elif eps := perturbation_conf["epsilon"]:
            eps_data_path = os.path.join(
                data_dir,
                dataset,
                f"{dataset}_eps_{eps}.npz",
            )

            if not os.path.isfile(eps_data_path):
                util_ref.log(
                    f"{self.__class__} perturbing data from {data_path} to {eps_data_path}",
                    1,
                )

                MotionSenseConditionalContainer.perturb(
                    data_path, eps_data_path, perturbation_conf["epsilon"]
                )
        else:
            eps_data_path = None

        super().__init__(util_ref, data_path, data_conf, eps_data_path, data_indices)

    def preprocess_data(self, model_type: str, dim_y: int = 100) -> None:
        """Prepare sensor data for specific model.
        here, the following steps are performed.
            1. Create One-hot encoding for labels.
            2. Reshape data.
        Args:
            model_type (str): Only for compability..
            dim_y (int, optional): Only for compability..
        """
        X_train, X_test, X_val, y_train, y_test, y_val = self.unravel()

        # One hot encoding
        y_train = to_categorical(y_train, num_classes=6)
        y_test = to_categorical(y_test, num_classes=6)
        y_val = to_categorical(y_val, num_classes=6)

        # Expected shape (samples, 12, 500)
        if not (X_train.shape[1] == 12) and not (X_train.shape[2] == 500):
            X_train = X_train.reshape(-1, 12, 500)

        if not (X_test.shape[1] == 12) and not (X_test.shape[2] == 500):
            X_test = X_test.reshape(-1, 12, 500)

        if not (X_val.shape[1] == 12) and not (X_val.shape[2] == 500):
            X_val = X_val.reshape(-1, 12, 500)

        self._set_data(X_train, X_test, X_val, y_train, y_test, y_val)

    @staticmethod
    def perturb(data_path: str, eps_data_path: str, epsilon: float) -> None:
        sensitivity = 2

        def add_laplace_noise(feature: np.array) -> np.array:
            noise_scale = sensitivity / epsilon
            noise = np.random.laplace(scale=noise_scale, size=len(feature))
            perturbed_feature = feature + noise
            # normalize feature between [-1,1]
            norm_perturbed_feature = np.interp(
                perturbed_feature,
                (perturbed_feature.min(), perturbed_feature.max()),
                (-1, +1),
            )
            return norm_perturbed_feature

        msDataset = MotionSenseConditionalContainer.get_MotionSense()
        data = msDataset.iloc[:, list(range(0, 13))]

        for i in range(0, 12):
            data.iloc[:, i] = add_laplace_noise(data.iloc[:, i])

        X_perturbed, y = MotionSenseConditionalContainer.createWindows_MotionSense(data)
        DataContainer.save_data(eps_data_path, X_perturbed, y)

    @staticmethod
    def get_data(data_path: str, dataset: str):
        start = datetime.now()
        msDataset = MotionSenseConditionalContainer.get_MotionSense()
        data = msDataset.iloc[:, list(range(0, 13))]
        data = MotionSenseConditionalContainer.normalize_MotionSense(data)
        X, y = MotionSenseConditionalContainer.createWindows_MotionSense(data)
        stop = datetime.now()
        print(f"\nOverall calculation took {stop - start}", flush=True)
        DataContainer.save_data(data_path, X, y)

    # Following methods are heavily based on: https://github.com/mmalekzadeh/motion-sense
    @staticmethod
    def get_MotionSense():
        ACT_LABELS = ["dws", "ups", "wlk", "jog", "std", "sit"]

        TRIAL_CODES = {
            ACT_LABELS[0]: [1, 2, 11],
            ACT_LABELS[1]: [3, 4, 12],
            ACT_LABELS[2]: [7, 8, 15],
            ACT_LABELS[3]: [9, 16],
            ACT_LABELS[4]: [6, 14],
            ACT_LABELS[5]: [5, 13],
        }
        sdt = ["attitude", "userAcceleration", "rotationRate", "gravity"]
        act_labels = ACT_LABELS[0:6]

        trial_codes = [TRIAL_CODES[act] for act in act_labels]
        dt_list = MotionSenseConditionalContainer.set_data_types(sdt)
        dataset = MotionSenseConditionalContainer.create_time_series(
            dt_list, act_labels, trial_codes, mode="raw", labeled=True
        )

        return dataset

    @staticmethod
    def set_data_types(data_types=["userAcceleration"]):
        """
        Select the sensors and the mode to shape the final dataset.

        Args:
            data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration]
        Returns:
            It returns a list of columns to use for creating time-series from files.
        """
        dt_list = []
        for t in data_types:
            if t != "attitude":
                dt_list.append([t + ".x", t + ".y", t + ".z"])
            else:
                dt_list.append([t + ".roll", t + ".pitch", t + ".yaw"])

        return dt_list

    @staticmethod
    def create_time_series(dt_list, act_labels, trial_codes, mode="mag", labeled=True):
        """
        Args:
            dt_list: A list of columns that shows the type of data we want.
            act_labels: list of activites
            trial_codes: list of trials
            mode: It can be "raw" which means you want raw data
            for every dimention of each data type,
            [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
            or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
            labeled: True, if we want a labeld dataset. False, if we only want sensor values.
        Returns:
            It returns a time-series of sensor data.

        """
        num_data_cols = len(dt_list) if mode == "mag" else len(dt_list * 3)

        if labeled:
            dataset = np.zeros(
                (0, num_data_cols + 7)
            )  # "7" --> [act, code, weight, height, age, gender, trial]
        else:
            dataset = np.zeros((0, num_data_cols))

        ds_list = MotionSenseConditionalContainer.get_ds_infos()

        for sub_id in ds_list["code"]:
            for act_id, act in enumerate(act_labels):
                for trial in trial_codes[act_id]:
                    fname = (
                        MOTIONSENSE_RAW_SENSOR_PATH
                        + act
                        + "_"
                        + str(trial)
                        + "/sub_"
                        + str(int(sub_id))
                        + ".csv"
                    )
                    raw_data = pd.read_csv(fname)
                    raw_data = raw_data.drop(["Unnamed: 0"], axis=1)
                    vals = np.zeros((len(raw_data), num_data_cols))
                    for x_id, axes in enumerate(dt_list):
                        if mode == "mag":
                            vals[:, x_id] = (raw_data[axes] ** 2).sum(axis=1) ** 0.5
                        else:
                            vals[:, x_id * 3 : (x_id + 1) * 3] = raw_data[axes].values
                        vals = vals[:, :num_data_cols]
                    if labeled:
                        lbls = np.array(
                            [
                                [
                                    act_id,
                                    sub_id - 1,
                                    ds_list["weight"][sub_id - 1],
                                    ds_list["height"][sub_id - 1],
                                    ds_list["age"][sub_id - 1],
                                    ds_list["gender"][sub_id - 1],
                                    trial,
                                ]
                            ]
                            * len(raw_data)
                        )
                        vals = np.concatenate((vals, lbls), axis=1)
                    dataset = np.append(dataset, vals, axis=0)
        cols = []
        for axes in dt_list:
            if mode == "raw":
                cols += axes
            else:
                cols += [str(axes[0][:-2])]

        if labeled:
            cols += ["act", "id", "weight", "height", "age", "gender", "trial"]

        dataset = pd.DataFrame(data=dataset, columns=cols)
        return dataset

    @staticmethod
    def get_ds_infos():
        """
        Read the file includes data subject information.

        Data Columns:
        0: code [1-24]
        1: weight [kg]
        2: height [cm]
        3: age [years]
        4: gender [0:Female, 1:Male]

        Returns:
            A pandas DataFrame that contains information about data subjects' attributes.
        """

        dss = pd.read_csv(MOTIONSENSE_RAW_SENSOR_PATH + "data_subjects_info.csv")

        return dss

    @staticmethod
    def normalize_MotionSense(df):
        """
        Normalize values between -1 and 1

        Returns:
            Normalized df
        """
        # Normalizing columns
        for i in range(0, 12):
            df.iloc[:, i] = np.interp(
                df.iloc[:, i], (df.iloc[:, i].min(), df.iloc[:, i].max()), (-1, +1)
            )
        return df

    @staticmethod
    def createWindows_MotionSense(Dataframe, SEGMENT_TIME_SIZE=500, TIME_STEP=20):
        """
        Create windows of size SEGMENT_TIME_SIZE.

        Returns:
            Time series windows and labels.
        """
        data = Dataframe.dropna()

        data_convoluted = []
        labels = []

        # Slide a "SEGMENT_TIME_SIZE" wide window with a step size of "TIME_STEP"
        for i in range(0, len(data) - SEGMENT_TIME_SIZE, TIME_STEP):
            gR = data["attitude.roll"].values[i : i + SEGMENT_TIME_SIZE]
            gP = data["attitude.pitch"].values[i : i + SEGMENT_TIME_SIZE]
            gY = data["attitude.yaw"].values[i : i + SEGMENT_TIME_SIZE]
            aX = data["userAcceleration.x"].values[i : i + SEGMENT_TIME_SIZE]
            aY = data["userAcceleration.y"].values[i : i + SEGMENT_TIME_SIZE]
            aZ = data["userAcceleration.z"].values[i : i + SEGMENT_TIME_SIZE]

            rX = data["rotationRate.x"].values[i : i + SEGMENT_TIME_SIZE]
            rY = data["rotationRate.y"].values[i : i + SEGMENT_TIME_SIZE]
            rZ = data["rotationRate.z"].values[i : i + SEGMENT_TIME_SIZE]
            grX = data["gravity.x"].values[i : i + SEGMENT_TIME_SIZE]
            grY = data["gravity.y"].values[i : i + SEGMENT_TIME_SIZE]
            grZ = data["gravity.z"].values[i : i + SEGMENT_TIME_SIZE]
            data_convoluted.append([gR, gP, gY, aX, aY, aZ, rX, rY, rZ, grX, grY, grZ])

            # Label for a data window is the label that appears most commonly
            label = stats.mode(data["act"][i : i + SEGMENT_TIME_SIZE])[0][0]

            labels.append(label)

        data_convoluted = np.asarray(data_convoluted, dtype=np.float32).transpose(
            0, 2, 1
        )

        return data_convoluted, labels
