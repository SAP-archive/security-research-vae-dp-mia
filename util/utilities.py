import datetime
import json
import math
import os
from typing import Any, Iterable, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from core.base import BaseClass
from tensorflow.keras.utils import to_categorical


class BaseHelper(object):
    """Helper class for basic tasks

    Can be used within a more sophisticated class to delegate repetitive tasks.

    """

    _instance: BaseClass = None

    def __init__(self, instance: BaseClass):
        """Constructor

        Args:
            instance (BaseClass): The instance this BaseHelper instance is serving as a helper for.
        """
        self._instance = instance

    @staticmethod
    def _read_config(
        config: dict, *args: str, strict: bool = True
    ) -> Union[Any, Iterable[Any]]:
        """Can be used to read values from a model config

        Has to be static to be callable before an actual helper instance is created.

        Args:
            config (dict): Config to read values from.
            *args: The keys of the values that should be read.
            strict (bool, optional): If True, an exception will be raised when a supplied key is missing in config.
                                     If False, None will be returned  for missing keys. Defaults to True.

        Raises:
            Exception: When strict is True and a value in args is not a valid key in config.

        Returns:
            Union[Any, Iterable[Any]]: Either a single value or a list of values.
        """

        values = list()

        for arg in args:

            if arg in config.keys():
                values.append(config.get(arg))
            elif not strict:
                values.append(None)
            else:
                raise Exception('Config is lacking required key "{0}"'.format(arg))

        if len(values) == 1:
            return values[0]
        else:
            return values

    def log(self, msg: str, required_level: int) -> None:
        """Log helper

        Args:
            msg (str): The message to print.
            required_level (int): The verbosity level that is required to print the message.

        """

        if self._instance._verbose >= required_level:
            print(msg)

    def config_valid(self, config: dict) -> Union[str, None]:
        """Validates config against base config

        Args:
            config (dict): A json/dict that represents the config.

        Returns:
            Union[str, None]: A string containing the reason the config is rejected. None if the config is valid.

        """

        # Load a base config according to the instance from which function was called
        template_name = "_".join([self._instance._type, "config_template.json"])
        with open(
            os.path.join(
                self._instance._base_path, self._instance._configs_dir, template_name
            ),
            "r",
        ) as f:
            template = json.loads(f.read())

        # Remove optional keys
        required_keys = [k for k in template.keys() if not k.startswith("?")]
        optional_keys = [
            k.replace("?", "") for k in template.keys() if k.startswith("?")
        ]

        # Check if all required keys are included
        if not set(required_keys).issubset(set(config.keys())):
            missing_keys = '", "'.join(
                set(set(required_keys)).difference(set(config.keys()))
            )
            return f'Required key(s) "{missing_keys}" missing in config'

        # Check if a value is supplied for required keys
        empty_fields = [
            k
            for k in required_keys
            if config[k] == "" or (config[k] == 0 and type(config[k]) != bool)
        ]
        if len(empty_fields) > 0:
            return f'No value(s) or value "0" supplied for keys(s) "{0}"'.format(
                '", "'.join(empty_fields)
            )

        # Check if optional keys are valid
        optional_keys_used = set(config.keys()).difference(required_keys)
        if not optional_keys_used.issubset(optional_keys):
            invalid_keys = '", "'.join(
                set(optional_keys_used).difference(set(optional_keys))
            )
            return 'Invalid key(s) "{0}" used in config'.format(invalid_keys)

        self.log(f"Config valid for {self._instance.__class__}", 1)
        return None

    def read_config(self, *args: str, strict: bool = True) -> Union[Any, Iterable[Any]]:
        """Can be used to read values from a model config

        Args:
            *args: The keys of the values that should be read.
            strict: If True, an exception will be raised when a supplied key is missing in config.
                    If False, None will be returned  for missing keys. Defaults to True.

        Returns:
            Union[Any, Iterable[Any]]: Either a single value or a list of values.

        """
        self.log(f'Read config values {", ".join(args)}', 2)
        values = self._read_config(self._instance._config, *args, strict=strict)
        return values

    def get_partial_config_dict(self, *args: str, strict: bool = True) -> dict:
        """Read Values from config and return them as a dict

        Args:
            *args: The keys of the values that should be read.
            strict: If True, an exception will be raised when a supplied key is missing in config.
                    If False, None will be returned for missing keys. Defaults to True.

        Returns:
            Dict: A dict with keys and the read values

        """
        self.log(f'Read config values {", ".join(args)}', 2)
        values = self._read_config(self._instance._config, *args, strict=strict)
        return dict(zip(args, values))


# Other utility functions


def has_quadratic_shape(data):
    if data.shape[1] == data.shape[2]:
        return True
    return False


def combine_images(generated_images):
    num_images = generated_images.shape[0]
    new_width = int(math.sqrt(num_images))
    new_height = int(math.ceil(float(num_images) / new_width))
    grid_shape = generated_images.shape[1:3]
    grid_image = np.zeros(
        (new_height * grid_shape[0], new_width * grid_shape[1]),
        dtype=generated_images.dtype,
    )
    for index, img in enumerate(generated_images):
        i = int(index / new_width)
        j = index % new_width
        grid_image[
            i * grid_shape[0] : (i + 1) * grid_shape[0],
            j * grid_shape[1] : (j + 1) * grid_shape[1],
        ] = img[:, :, 0]
    return grid_image


def generate_noise(shape: tuple):
    noise = np.random.default.rng().uniform(0, 1, size=shape)
    return noise


def generate_condition_embedding(label: int, nb_of_label_embeddings: int):
    label_embeddings = np.zeros((nb_of_label_embeddings, 100))
    label_embeddings[:, label] = 1
    return label_embeddings


def generate_images(generator, nb_images: int, label: int):
    noise = generate_noise((nb_images, 100))
    label_batch = generate_condition_embedding(label, nb_images)
    generated_images = generator.predict([noise, label_batch], verbose=0)
    return generated_images


def generate_mnist_image_grid(
    generator, dim_z: int, dim_y: int, classes: list, title: str = "Generated images"
):
    generated_images = []

    if classes is None:
        classes = range(10)

    for i in classes:
        noise = generate_noise((10, dim_z))
        # label_input = generate_condition_embedding(i, 10)
        label_input = to_categorical([i] * 10, dim_y)
        # label_input = np.array([i] * 10)
        # print(label_input.shape)
        gen_images = generator.predict([noise, label_input], verbose=0)
        generated_images.extend(gen_images)

    generated_images = np.array(generated_images)
    image_grid = combine_images(generated_images)
    image_grid = inverse_transform_images(image_grid)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(image_grid, cmap="gray")
    ax.set_title(title)
    fig.canvas.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return image


def save_generated_image(image, epoch, iteration, folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    file_path = "{0}/{1}_{2}.png".format(folder_path, epoch, iteration)
    cv2.imwrite(file_path, image.astype(np.uint8))


def transform_images_minus_one_one_range(images: np.ndarray):
    """
    Transform images to [-1, 1]
    """

    images = (images.astype(np.float32) - 127.5) / 127.5
    return images


def inverse_transform_images_minus_one_one_range(images: np.ndarray):
    """
    From the [-1, 1] range transform the images back to [0, 255]
    """

    images = images * 127.5 + 127.5
    images = images.astype(np.uint8)
    return images


def transform_images_zero_one_range(images: np.ndarray):
    """
    Transform images to [0, 1]
    """

    images = images.astype(np.float32) / 255.0
    return images


def inverse_transform_images_zero_one_range(images: np.ndarray):
    """
    From the [0, 1] range transform the images back to [0, 255]
    """

    images = images * 255.0
    images = images.astype(np.uint8)
    return images


"""
def generate_mnist_image_grid_vae(decoder, dim_z: int, dim_y: int, epoch: int, title: str = "Generated images"):
    generated_images = []

    for i in range(10):
        noise = generate_noise((10, dim_z))
        # label_input = generate_condition_embedding(i, 10)
        label_input = to_categorical([i] * 10, dim_y)
        # label_input = np.array([i] * 10)
        # print(label_input.shape)
        gen_images = decoder.predict(np.concatenate([noise, label_input], axis=1), verbose=0)
        gen_images = np.reshape(gen_images, (10, 28, 28, 1))
        generated_images.extend(gen_images)

    generated_images = np.array(generated_images)
    image_grid = combine_images(generated_images)
    image_grid = inverse_transform_images(image_grid)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(image_grid, cmap="gray")
    ax.set_title(title)
    fig.canvas.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    #return image
    save_generated_image(image, epoch, 0, "./images/generated_per_epoch")

    pic_num = 0
    variations = 30 # rate of change; higher is slower
    for j in range(dim_z, dim_z + dim_y - 1):
        for k in range(variations):
            v = np.zeros((1, dim_z + dim_y))
            v[0, j] = 1 - (k/variations)
            v[0, j+1] = (k/variations)
            generated = decoder.predict(v)
            pic_idx = j - dim_z + (k/variations)
            file_name = './images/transition_50/img{0:.3f}.jpg'.format(pic_idx)
            cv2.imwrite(file_name, generated.reshape((28,28)))
            pic_num += 1
"""


def generate_mnist_image_grid_vae(models, test_data, epoch, dim_z, batch_size=128):

    if dim_z != 2:
        print("Grid works only for dim_z = 2")
        return None

    encoder, decoder = models
    x_test, y_test = test_data
    # os.makedirs(model_name, exist_ok=True)

    filename = "./images/generated_per_epoch/vae_mean_epoch_" + str(epoch) + ".png"
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict([x_test, y_test], batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test.argmax(axis=1))
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    # plt.show()

    filename = (
        "./images/generated_per_epoch/digits_over_latent_epoch_" + str(epoch) + ".png"
    )
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(
                np.concatenate([z_sample, np.array([y_test[0]])], axis=1)
            )
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig(filename)
    # plt.show()


def check_if_list_and_matches_length(
    param: "object", expected_length: int, name: str = "parameter"
) -> list:
    """
        Checks, if the passed variable is a list. Otherwise creates a list of the variable of length expected_length.
        Then checks if the length of the list is the expected length.

    Args:
        param (object): The variable to be checked.
        num_layers (int): number of elements expected in the list.
        name (str, optional): Name of the parameter used in raised Error. Defaults to "parameter".

    Raises:
        ValueError: Length of the list does not match expected_length.

    Returns:
        list: List of the passed variable or the list itself
    """

    if not (isinstance(param, list) or isinstance(param, np.ndarray)):
        param = [param] * expected_length

    if len(param) != expected_length:
        raise ValueError(
            f"Passed {name} ({param}) does not match number of expected layers ({expected_length})"
        )

    return param


class CustomJSONEncoder(json.JSONEncoder):
    """
    JSON cannot serialize numpy datatypes or other custom datatypes out of the box.
    Thus, we use and extend a custom JSON encoder based on https://github.com/hmallen/numpyencoder/blob/f8199a61ccde25f829444a9df4b21bcb2d1de8f2/numpyencoder/numpyencoder.py
    """

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()

        return json.JSONEncoder.default(self, obj)
