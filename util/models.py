import os
from abc import abstractmethod
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, layers, losses, models, optimizers
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow_privacy.privacy.analysis.rdp_accountant import (
    compute_rdp, get_privacy_spent)

import util.figures as figure_helper
import util.utilities as util_helper


def sampling(args):
    """Custom Sampling Layer for VAE. Uses reparameterization trick by sampling from an isotropic unit Gaussian (Kingam & Welling - Auto-Encoding Variational Bayes, 2014)

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return tf.add(z_mean, tf.multiply(K.exp(tf.multiply(0.5, z_log_var)), epsilon))


def custom_scaled_tanh(x):
    """Custom scaled tanh for DP-VAE approach."""
    scale = 3
    return activations.tanh(x) * scale


def sigma_bound(lower_bound):
    """
    Custom activation function to bound z_log_sigma.

    Args:
        lower_bound (float): Lower bound of z_log_sigma.
    """

    def custom_sigma_bound(x):
        lower_bound_tf32 = tf.dtypes.cast(lower_bound, tf.float32)
        return tf.math.maximum(x, tf.math.log(lower_bound_tf32))

    return custom_sigma_bound


def get_deep_conv(
    name: str,
    num_layers: int,
    inp: "layers.Input",
    filters: list or int,
    kernels: list or int or tuple,
    stride: list or int,
    batch_norm: list or bool = True,
    activation: list or bool = True,
) -> "tf.tensor":

    filters = util_helper.check_if_list_and_matches_length(
        filters, num_layers, "filters"
    )
    kernels = util_helper.check_if_list_and_matches_length(
        kernels, num_layers, "kernels"
    )
    stride = util_helper.check_if_list_and_matches_length(stride, num_layers, "stride")
    batch_norm = util_helper.check_if_list_and_matches_length(
        batch_norm, num_layers, "batch_norm"
    )
    activation = util_helper.check_if_list_and_matches_length(
        activation, num_layers, "activation"
    )

    x = inp

    for idx, (f, k, s, bn, act) in enumerate(
        zip(filters, kernels, stride, batch_norm, activation)
    ):
        x = layers.Conv2D(
            filters=f,
            kernel_size=k,
            strides=s,
            activation=None,
            padding="same",
            name=f"{name}_conv2D_{idx}",
        )(x)
        if bn:
            x = layers.BatchNormalization(name=f"{name}_BatchNorm_{idx}")(x)
        if act:
            x = layers.LeakyReLU(alpha=0.1, name=f"{name}_LeakyReLU_{idx}")(x)

    return x


def get_deep_conv_with_upsampling(
    name: str,
    num_layers: int,
    inp: "layers.Input",
    upsampling_size: list or int or tuple,
    filters: list or int,
    kernels: list or int or tuple,
    stride: list or int,
    batch_norm: list or bool = True,
    activation: list or bool = True,
    output_layer: bool = True,
) -> "tf.tensor":

    upsampling_size = util_helper.check_if_list_and_matches_length(
        upsampling_size, num_layers, "upsampling_size"
    )
    filters = util_helper.check_if_list_and_matches_length(
        filters, num_layers, "filters"
    )
    kernels = util_helper.check_if_list_and_matches_length(
        kernels, num_layers, "kernels"
    )
    stride = util_helper.check_if_list_and_matches_length(stride, num_layers, "stride")
    batch_norm = util_helper.check_if_list_and_matches_length(
        batch_norm, num_layers, "batch_norm"
    )
    activation = util_helper.check_if_list_and_matches_length(
        activation, num_layers, "activation"
    )

    x = inp

    for idx, (ups, f, k, s, bn, act) in enumerate(
        zip(upsampling_size, filters, kernels, stride, batch_norm, activation)
    ):
        x = layers.UpSampling2D(
            ups,
            name=f"{name}_UpSampling2D_{idx}",
        )(x)
        x = layers.Conv2D(
            filters=f,
            kernel_size=k,
            strides=s,
            activation=None,
            padding="same",
            name=f"{name}_conv2D_{idx}",
        )(x)
        if bn:
            x = layers.BatchNormalization(name=f"{name}_BatchNorm_{idx}")(x)
        if act:
            x = layers.LeakyReLU(alpha=0.1, name=f"{name}_LeakyReLU_{idx}")(x)

    if output_layer:
        x = layers.Activation("sigmoid")(x)

    return x


def compute_epsilon_delta_with_rdp(
    epochs: int, batch_size: int, noise_multiplier: float, num_train_samples: int
) -> Tuple[float, float]:
    """
        Computes the spent privacy budget for the cdp training.

    Args:
        epochs (int): Number of epochs during training.
        batch_size (int): Batch size of training.
        noise_multiplier (float): Noise Multiplier of DP Optimizer.
        num_train_samples (int): Size of the training set.

    Returns:
        Tuple[float, float]: Computed epsilon and delta, where delta is 1 / num_train_samples.
    """

    if noise_multiplier == 0.0 or noise_multiplier is None:
        return (float("inf"), None)

    steps = epochs * num_train_samples / batch_size
    sampling_probability = batch_size / num_train_samples
    target_delta = 1 / num_train_samples

    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    rdp = compute_rdp(
        q=sampling_probability,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=orders,
    )

    epsilon = get_privacy_spent(orders, rdp, target_delta=target_delta)[0]

    return (epsilon, target_delta)


def compute_epsilon_delta_for_ldp(
    noise_bound: int or float, latent_dim: int, num_samples: int
) -> Tuple[float, float]:
    """Compute spent privacy for VAE as LDP generator.

    Args:
        noise_bound (int or float): Noise bound enforced on sigma.
        latent_dim (int): Size  of the latent dimension.
        num_samples (int): Number of samples passed through vae.

    Returns:
        Tuple[float, float]: epsilon & delta value.
    """

    if noise_bound == 0.0 or noise_bound is None:
        return (float("inf"), None)

    scale = 3
    sensitivity = 2 * scale * np.sqrt(latent_dim)

    target_delta = 1 / num_samples
    epsilon = (sensitivity * np.sqrt(2 * np.log(1.25 / target_delta))) / noise_bound

    return (epsilon, target_delta)


# ----- CUSTOM keras.Model Classes -----
class GradientModel(models.Model):
    """custom subclass of tensorflow model to save median and mean per step gradients to a file."""

    def __init__(
        self, *args, gradient_file: str = "/tmp/gradient_file", **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gradient_file = "".join(["file://", gradient_file])

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        abs_gradients = tf.math.abs(
            tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)
        )

        tf.print(abs_gradients, output_stream=self.gradient_file, summarize=-1)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


class MemoryEfficientGradientModel(GradientModel):
    """drop accuracy, especially for median score to achieve great performance boost."""

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        abs_gradients = tf.math.abs(
            tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)
        )

        per_step_median = tfp.stats.percentile(
            abs_gradients, 0.5, interpolation="midpoint"
        )
        per_step_mean = tf.math.reduce_mean(abs_gradients)

        tf.print(
            per_step_median, per_step_mean, output_stream=self.gradient_file, sep="\t"
        )

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


# ----- CUSTOM MODELS -----
class customClassifier:
    def __init__(
        self,
        optimizer: "keras.optimizers" = None,
        loss: "keras.losses" = None,
        metrics: list = None,
    ) -> None:
        self.model = None
        self.optimizer = optimizer if optimizer is not None else optimizers.Adam()
        self.loss = (
            loss
            if loss is not None
            else losses.CategoricalCrossentropy(from_logits=False)
        )
        self.metrics = metrics if metrics is not None else ["categorical_accuracy"]

    def fit(
        self,
        train_generator,
        epochs: int = 10,
        batch_size: int = 32,
        val_generator=None,
    ) -> None:
        if not self.model:
            raise RuntimeWarning(
                f"{self.__class__.__name__} couldn't train. Create model first."
            )

        self._history = self.model.fit(
            train_generator,
            epochs=epochs,
            steps_per_epoch=len(train_generator),
            validation_data=val_generator,
            validation_steps=len(val_generator) if val_generator else None,
        )

    def evaluate(self, data_generator) -> list:
        return self.model.evaluate(data_generator, steps=len(data_generator))

    def full_evaluation(self, data_generator) -> dict:
        prediction = self.model.predict(data_generator)

        y_hat = np.argmax(prediction, axis=1)
        y = np.argmax(data_generator.y, axis=1)

        prec, rec, f1, weights = metrics.precision_recall_fscore_support(y, y_hat)
        acc = metrics.accuracy_score(y, y_hat)
        cm = metrics.confusion_matrix(y, y_hat)

        return {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "weights": weights,
            "accuracy": acc,
            "confusion_matrix": cm,
        }

    @abstractmethod
    def create(self) -> None:
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )


class customVGG16Model(customClassifier):
    def create(
        self,
        input_shape: tuple = (64, 64, 3),
        num_classes: int = 100,
        hidden_dim: int = 4096,
        path_to_weights: str = "./custom_models/rcmalli_vggface_tf_notop_vgg16.h5",
    ) -> "models.Model":
        img_input = layers.Input(shape=input_shape)
        img_reshape = tf.image.resize(img_input, [224, 224])

        # Block 1
        x = layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same",
            name="conv1_1",
            trainable=False,
        )(img_reshape)
        x = layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same",
            name="conv1_2",
            trainable=False,
        )(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool1", trainable=False)(
            x
        )

        # Block 2
        x = layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same",
            name="conv2_1",
            trainable=False,
        )(x)
        x = layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same",
            name="conv2_2",
            trainable=False,
        )(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool2", trainable=False)(
            x
        )

        # Block 3
        x = layers.Conv2D(
            256,
            (3, 3),
            activation="relu",
            padding="same",
            name="conv3_1",
            trainable=False,
        )(x)
        x = layers.Conv2D(
            256,
            (3, 3),
            activation="relu",
            padding="same",
            name="conv3_2",
            trainable=False,
        )(x)
        x = layers.Conv2D(
            256,
            (3, 3),
            activation="relu",
            padding="same",
            name="conv3_3",
            trainable=False,
        )(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool3", trainable=False)(
            x
        )

        # Block 4
        x = layers.Conv2D(
            512,
            (3, 3),
            activation="relu",
            padding="same",
            name="conv4_1",
            trainable=False,
        )(x)
        x = layers.Conv2D(
            512,
            (3, 3),
            activation="relu",
            padding="same",
            name="conv4_2",
            trainable=False,
        )(x)
        x = layers.Conv2D(
            512,
            (3, 3),
            activation="relu",
            padding="same",
            name="conv4_3",
            trainable=False,
        )(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool4", trainable=False)(
            x
        )

        # Block 5
        x = layers.Conv2D(
            512,
            (3, 3),
            activation="relu",
            padding="same",
            name="conv5_1",
            trainable=False,
        )(x)
        x = layers.Conv2D(
            512,
            (3, 3),
            activation="relu",
            padding="same",
            name="conv5_2",
            trainable=False,
        )(x)
        x = layers.Conv2D(
            512,
            (3, 3),
            activation="relu",
            padding="same",
            name="conv5_3",
            trainable=False,
        )(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool5", trainable=False)(
            x
        )

        tmp = models.Model(img_input, x)
        tmp.load_weights(path_to_weights, by_name=True)

        # custom last layers which can be finetuned
        x = layers.Flatten(name="flatten")(tmp.get_layer("pool5").output)
        x = layers.Dense(hidden_dim, activation="relu", name="fc6")(x)
        x = layers.Dense(hidden_dim, activation="relu", name="fc7")(x)
        out = layers.Dense(num_classes, activation="softmax", name="fc8")(x)
        model = models.Model(tmp.input, out)

        del tmp

        self.model = model

        super().create()


class customHARCNNModel(customClassifier):
    """Custom implementation for human activity recognition with CNN based on  https://github.com/Shahnawax/HAR-CNN-Keras"""

    def create(
        self, input_shape: tuple = (12, 500), num_classes: int = 6
    ) -> "models.Model":

        x = layers.Input(shape=input_shape)
        conv1 = layers.Conv1D(48, 1, strides=1, activation="relu")(x)
        dropout1 = layers.Dropout(0.3)(conv1)
        conv2 = layers.Conv1D(96, 2, strides=1, activation="relu")(dropout1)
        dropout2 = layers.Dropout(0.3)(conv2)
        conv3 = layers.Conv1D(128, 1, strides=1, activation="relu")(dropout2)
        dropout3 = layers.Dropout(0.3)(conv3)
        globalMaxPool = layers.GlobalMaxPooling1D()(dropout3)
        dense1 = layers.Dense(64, activation="sigmoid")(globalMaxPool)
        dense2 = layers.Dense(num_classes, activation="softmax")(dense1)
        self.model = models.Model(inputs=x, outputs=dense2)

        super().create()


# ----- CUSTOM CALLBACKS -----
class customLDACallback(callbacks.Callback):
    def __init__(
        self,
        encoder: "keras.Model",
        data: list,
        labels: list or None,
        batch_size: int,
        figure_dir: str,
        epoch_modulo: int = 5,
        plot_mean: bool = True,
        plot_sigma: bool = False,
        plot_latent: bool = True,
    ) -> None:
        """Custom Callback for better latent space analysis.

        Args:
            encoder (keras.Model): Encoder to create latent space. Expected to return [mu, sigma, z].
            data (list or generator): The data to run through the encoder. Either a list or a generator.
            labels (list or None): Corresponding labels for data. If None, we assume data to be a generator and get labels from there.
            batch_size (int): Batch size to use for `encoder.predict`
            figure_dir (str): Path to save the figures to.
            epoch_modulo (int, optional): How often the callback should plot the latent space. Defaults to 5.
            plot_mean (bool, optional): Whether to plot analysis for mean. Defaults to True.
            plot_sigma (bool, optional): Whether to plot analysis for sigma. Defaults to False.
            plot_latent (bool, optional): Whether to plot analysis for z. Defaults to True.
        """
        self.encoder = encoder
        self.data = data
        self.batch_size = batch_size
        self.figure_dir = figure_dir
        self.epoch_modulo = epoch_modulo
        self.plot_mean = plot_mean
        self.plot_sigma = plot_sigma
        self.plot_latent = plot_latent

        self.check_labels(labels)
        self.check_if_active()

    def check_if_active(self) -> None:
        """Checks whether any flag was set to true. Otherwise raises a RuntimeWarning."""
        if not self.plot_mean and not self.plot_sigma and not self.plot_latent:
            raise RuntimeWarning(
                f"{self.__class__.__name__} set up without plotting any figures, thus, it will only make training slower."
            )

    def check_labels(self, labels) -> None:
        """Checks whether `labels` where passed. otherwise assumes that `data` is a generator and takes the labels from `generator.__getitems__(n)[-1]`."""
        if labels is not None:
            self.labels = labels
        else:
            labels = []
            for n in range(len(self.data)):
                labels.append(self.data.__getitem__(n)[-1])
            self.labels = np.concatenate(labels)

    def performLDA(self, x, savepath):
        """Performs LDA analysis on `x` and saves a plot to `savepath`."""
        x = np.clip(x, np.finfo(np.float32).min + 1, np.finfo(np.float32).max - 1)
        y = np.argmax(self.labels, axis=1)
        clf = LinearDiscriminantAnalysis(n_components=2)
        X_lda = clf.fit(x, y).transform(x)
        figure_helper.scatter_2d_data(X_lda, y=y, savepath=savepath)

    def check_and_run(self, epoch):
        mean, sigma, z = self.encoder.predict(self.data, batch_size=self.batch_size)

        if self.plot_mean:
            try:
                self.performLDA(
                    mean, os.path.join(self.figure_dir, f"LDA_mean_{epoch}.pdf")
                )
            except Exception as e:
                print(e)

        if self.plot_sigma:
            try:
                self.performLDA(
                    sigma, os.path.join(self.figure_dir, f"LDA_sigma_{epoch}.pdf")
                )
            except Exception as e:
                print(e)

        if self.plot_latent:
            try:
                self.performLDA(
                    z, os.path.join(self.figure_dir, f"LDA_latent_{epoch}.pdf")
                )
            except Exception as e:
                print(e)

    def on_epoch_end(self, epoch, logs=None) -> None:
        """Runs an LDA analysis if the `epoch` % epoch_modulo is 0."""
        if (epoch % self.epoch_modulo) == 0:
            self.check_and_run(epoch)

    def on_train_end(self, logs=None) -> None:
        """Runs LDA one last time after training and plots the final analysis."""
        self.check_and_run("train_end")


class customPCACallback(callbacks.Callback):
    def __init__(
        self,
        encoder: "keras.Model",
        data: list,
        labels: list or None,
        batch_size: int,
        figure_dir: str,
        epoch_modulo: int = 5,
        plot_mean: bool = True,
        plot_sigma: bool = False,
        plot_latent: bool = True,
    ) -> None:
        """Custom Callback for better latent space analysis.

        Args:
            encoder (keras.Model): Encoder to create latent space. Expected to return [mu, sigma, z].
            data (list or generator): The data to run through the encoder. Either a list or a generator.
            labels (list or None): Corresponding labels for data. Only used for plotting. If None, we assume data to be a generator and check for labels there.
            batch_size (int): Batch size to use for `encoder.predict`
            figure_dir (str): Path to save the figures to.
            epoch_modulo (int, optional): How often the callback should plot the latent space. Defaults to 5.
            plot_mean (bool, optional): Whether to plot analysis for mean. Defaults to True.
            plot_sigma (bool, optional): Whether to plot analysis for sigma. Defaults to False.
            plot_latent (bool, optional): Whether to plot analysis for z. Defaults to True.
        """
        self.encoder = encoder
        self.data = data
        self.batch_size = batch_size
        self.figure_dir = figure_dir
        self.epoch_modulo = epoch_modulo
        self.plot_mean = plot_mean
        self.plot_sigma = plot_sigma
        self.plot_latent = plot_latent

        self.check_labels(labels)
        self.check_if_active()

    def check_if_active(self) -> None:
        """Checks whether any flag was set to true. Otherwise raises a RuntimeWarning."""
        if not self.plot_mean and not self.plot_sigma and not self.plot_latent:
            raise RuntimeWarning(
                f"{self.__class__.__name__} set up without plotting any figures, thus, it will only make training slower."
            )

    def check_labels(self, labels) -> None:
        """Checks whether `labels` where passed. Otherwise assumes that `data` is a generator and checks whether labels are present in `generator.__getitems__(n)`."""
        if labels is not None:
            if len(labels.shape) > 1:
                labels = np.argmax(labels, axis=1)
            self.labels = labels
        else:
            data = self.data.__getitem__(0)
            if len(data) < 2:
                self.labels = None
                return

            labels = []
            for n in range(len(self.data)):
                labels.append(self.data.__getitem__(n)[-1])

            labels = np.concatenate(labels)

            if len(labels.shape) > 1:
                labels = np.argmax(labels, axis=1)

            self.labels = labels

    def performPCA(self, x, savepath):
        """Performs PCA on `x` and saves a plot to `savepath`."""
        x = np.clip(x, np.finfo(np.float32).min + 1, np.finfo(np.float32).max - 1)
        clf = PCA(n_components=2)
        X_lda = clf.fit(x).transform(x)
        figure_helper.scatter_2d_data(X_lda, self.labels, savepath=savepath)

    def check_and_run(self, epoch) -> None:
        mean, sigma, z = self.encoder.predict(self.data, batch_size=self.batch_size)

        if self.plot_mean:
            try:
                self.performPCA(
                    mean, os.path.join(self.figure_dir, f"PCA_mean_{epoch}.pdf")
                )
            except Exception as e:
                print(e)

        if self.plot_sigma:
            try:
                self.performPCA(
                    sigma, os.path.join(self.figure_dir, f"PCA_sigma_{epoch}.pdf")
                )
            except Exception as e:
                print(e)

        if self.plot_latent:
            try:
                self.performPCA(
                    z, os.path.join(self.figure_dir, f"PCA_latent_{epoch}.pdf")
                )
            except Exception as e:
                print(e)

    def on_epoch_end(self, epoch, logs=None) -> None:
        """Runs a PCA if the `epoch` % epoch_modulo is 0."""
        if (epoch % self.epoch_modulo) == 0:
            self.check_and_run(epoch)

    def on_train_end(self, logs=None) -> None:
        """Runs PCA one last time after training and plots the final analysis."""
        self.check_and_run("train_end")
