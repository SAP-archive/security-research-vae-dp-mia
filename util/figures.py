import itertools

import matplotlib.pyplot as plt
import numpy as np

from util import utilities as util_helper

CMAP_NAME = "Dark2"
CONTINOUS_CMAP_NAME = "summer"


def plot_pairwise_image_grid(
    A: list, B: list, columns: int = 5, savepath: str = None, show: bool = False
) -> None:
    """Plots a grid of images. Takes two lists, zips them together and plots them underneath each other.

    Args:
        A ([type]): First list of images.
        B ([type]): Second list of images.
        columns (int, optional): How many columns the grid should have. Defaults to 5.
        savepath ([type], optional): Where to save the image. Defaults to None.
    """

    num_pictures = len(A)
    rows = 2 * int(np.ceil(num_pictures / columns))

    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(columns, rows))
    for idx in range(num_pictures):
        axes[2 * (idx // columns), idx % columns].imshow(A[idx])
        axes[2 * (idx // columns) + 1, idx % columns].imshow(B[idx])

    for ax in axes.flatten():
        ax.set_axis_off()

    fig.tight_layout()
    fig.subplots_adjust(wspace=1e-5, hspace=1e-5)
    if savepath:
        fig.savefig(savepath.format(rows, columns))
    if show:
        plt.show(fig)
    plt.close(fig)


def plot_image_grid(A: list, savepath: str = None, show: bool = False) -> None:
    """Plots a grid of images. Takes a lists and plots the images.

    Args:
        A ([type]): First list of images.
        B ([type]): Second list of images.
        columns (int, optional): How many columns the grid should have. Defaults to 5.
        savepath ([type], optional): Where to save the image. Defaults to None.
    """

    num_pictures = len(A)
    rows = columns = int(np.ceil(np.sqrt(num_pictures)))

    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(columns, rows))
    for idx in range(num_pictures):
        axes[(idx // columns), idx % columns].imshow(A[idx])

    for ax in axes.flatten():
        ax.set_axis_off()

    fig.tight_layout()
    fig.subplots_adjust(wspace=1e-5, hspace=1e-5)
    if savepath:
        fig.savefig(savepath.format(rows, columns))
    if show:
        plt.show(fig)
    plt.close(fig)


def plot_labeled_image_comparison(
    A: list,
    names: str or list,
    columns: int = 5,
    savepath: str = None,
    show: bool = False,
) -> None:

    num_pictures = len(A)
    rows = num_pictures // columns

    util_helper.check_if_list_and_matches_length(names, rows, "names")

    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(columns, rows))

    for idx in range(num_pictures):
        axes[(idx // columns), idx % columns].imshow(A[idx])

    for r in range(rows):
        axes[r][0].set_title(names[r], rotation="vertical", x=-0.1, y=0.3)

    for ax in axes.flatten():
        ax.set_axis_off()

    fig.tight_layout()
    fig.subplots_adjust(wspace=1e-5, hspace=1e-5)
    if savepath:
        fig.savefig(savepath.format(rows, columns))
    if show:
        plt.show(fig)
    plt.close(fig)


def plot_values_over_keys(
    data: dict,
    xlabel: str = "epochs",
    ylabel: str = "score",
    fmt: str = "-",
    savepath: str = None,
    show: bool = False,
):
    ylogscale, ymaxval = False, 1e4

    num_keys = len(data)
    cmap = plt.get_cmap(CMAP_NAME)
    colors = cmap(np.flip(np.linspace(0, 0.9, num_keys)))

    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, key in enumerate(data.keys()):
        if isinstance(data[key], dict):
            ys = np.array([float(v) for v in data[key].values()])
            if np.max(ys) > ymaxval:
                ylogscale = True
            ax.plot(
                np.array([int(k) for k in data[key].keys()]),
                ys,
                "-",
                label=str(key),
                color=colors[idx],
            )
        else:
            ys = np.array([float(v) for v in data[key]])
            if np.max(ys) > ymaxval:
                ylogscale = True
            ax.plot(
                np.array(np.arange(len(data[key]))),
                ys,
                "-",
                label=str(key),
                color=colors[idx],
            )

    ax.set(xlabel=xlabel, ylabel=ylabel)
    if ylogscale:
        ax.set(yscale="log")
    ax.grid(ls="dashed", alpha=0.5)
    fig.tight_layout()
    fig.legend(ncol=min(4, len(data.keys())), loc=8)
    fig.subplots_adjust(bottom=0.2)
    if savepath:
        fig.savefig(savepath)
    if show:
        plt.show(fig)
    plt.close(fig)


def plot_data_as_histogram(
    data: list,
    xlabel: str = "label",
    ylabel: str = "# occurences",
    show_xticklabels: bool = False,
    savepath: str = None,
    show: bool = False,
) -> None:
    """Plots a histogram of the unique values in the provided data.

    Args:
        data (list): Data to plot.
        xlabel (str, optional): Label for x-axis. Defaults to "label".
        ylabel (str, optional): Label for y-axis. Defaults to "# occurences".
        show_xticklabels (bool, option): Whether to show xticklabels. Defaults to False.
        savepath (str, optional): Full path to save data to. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to False.
    """

    cmap = plt.get_cmap(CMAP_NAME)
    colors = cmap(np.flip(np.linspace(0, 0.9, 1)))

    num_labels = len(np.unique(data))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        data,
        bins=num_labels,
        rwidth=0.95,
        align="right",
        alpha=0.7,
        edgecolor="white",
        color=colors[0],
    )

    ax.set(xlabel=xlabel, ylabel=ylabel)
    if not show_xticklabels:
        ax.set(xticklabels="", xticks=[])

    ax.grid(ls="dashed", alpha=0.5)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath)
    if show:
        plt.show(fig)
    plt.close(fig)


def scatter_2d_data(
    data: list,
    y: list = None,
    xlabel: str = "x",
    ylabel: str = "y",
    savepath: str = None,
    show: bool = False,
) -> None:
    """Scatters provided data and saves the plot.

    Args:
        data (list): Data to be scattered. Expects shape (None, 2), where [:, 0] is x and [:, 1] is y.
        y (list, optional): Optional labels for data points. Defaults to None.
        xlabel (str, optional): Label for x-axis. Defaults to "x".
        ylabel (str, optional): Label for y-axis. Defaults to "y".
        savepath (str, optional): Full path to save data to. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to False.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    sca = ax.scatter(
        data[:, 0], data[:, 1], c=y if y is not None else [0], alpha=0.7, cmap=CMAP_NAME
    )

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(ls="dashed", alpha=0.5)
    fig.tight_layout()

    if y is not None:
        labels = np.unique(y)
        if (len_labels := len(labels)) <= 20:
            handles, _ = sca.legend_elements()
            fig.legend(
                handles,
                labels,
                ncol=min(5, len_labels),
                loc=8,
            )
            num_rows = int(np.ceil(len_labels / 5))
            fig.subplots_adjust(bottom=0.1 + 0.05 * num_rows)

    if savepath:
        fig.savefig(savepath)
    if show:
        plt.show(fig)
    plt.close(fig)


def plot_gradient_distribution(
    results: list, savepath: str = None, show: bool = False
) -> None:
    """Takes the output of the evaluate_gradient_file and plots how gradients change over the course of the training.

    Args:
        results (list): results of evaluate_gradient_file.
        savepath (str, optional): Full path to save data to. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to False.
    """
    cmap = plt.get_cmap(CMAP_NAME)
    colors = cmap(np.flip(np.linspace(0, 0.9, 2)))

    fig, ax = plt.subplots(figsize=(10, 5))
    steps = np.arange(results.shape[0])

    lower_err = np.abs(results[:, 0] - results[:, 2])
    upper_err = np.abs(results[:, 1] - results[:, 2])

    ax.scatter(steps, results[:, 3], marker="x", c=colors[0], s=1)
    ax.errorbar(
        steps,
        results[:, 2],
        yerr=np.vstack([lower_err, upper_err]),
        fmt=".",
        ecolor="red",
        elinewidth=0.3,
        ms=1,
        c=colors[1],
    )

    ax.set(xlabel="steps", ylabel="gradient", yscale="log")
    ax.grid(ls="dashed", alpha=0.5)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath)
    if show:
        plt.show(fig)
    plt.close(fig)


def plot_gradient_count(
    x: list, y: list, savepath: str = None, show: bool = False
) -> None:
    """Plots the count of different gradient values over the course of the training.

    Args:
        x (list): List of values of gradients.
        y (list): List of corresponding counts for gradients.
        savepath (str, optional): Full path to save data to. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to False.
    """

    cmap = plt.get_cmap(CMAP_NAME)
    colors = cmap(np.flip(np.linspace(0, 0.9, 1)))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, y, ".", ms=1, color=colors[0])

    ax.set(xlabel="value", ylabel="count", yscale="log")
    ax.grid(ls="dashed", alpha=0.5)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath)
    if show:
        plt.show(fig)
    plt.close(fig)


def plot_curve(
    x: list,
    y: list,
    label: str,
    xlabel: str,
    ylabel: str,
    baseline: list = None,
    savepath: str = None,
    show: bool = False,
) -> None:

    if not baseline:
        baseline = [[0, 1], [0, 1]]

    cmap = plt.get_cmap(CMAP_NAME)
    colors = cmap(np.flip(np.linspace(0, 0.9, 2)))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, y, label=label, linestyle="-", color=colors[1])
    ax.plot(*baseline, linestyle="--", color=colors[0])
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=(-0.05, 1.05),
        ylim=(-0.05, 1.05),
    )

    ax.grid(ls="dashed", alpha=0.5)
    fig.tight_layout()
    fig.legend(loc=8)
    fig.subplots_adjust(bottom=0.2)

    if savepath:
        fig.savefig(savepath)
    if show:
        plt.show(fig)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    normalize: bool = False,
    savepath: str = None,
    show: bool = False,
) -> None:
    """[summary]
    Based on https://sklearn.org/auto_examples/model_selection/plot_confusion_matrix.html.

    Args:
        cm (np.ndarray): [description]
        classes (list): [description]
        normalize (bool, optional): [description]. Defaults to False.
        savepath (str, optional): [description]. Defaults to None.
        show (bool, optional): [description]. Defaults to False.
    """

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    num_classes = len(classes)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=CONTINOUS_CMAP_NAME)

    fmt = ".2f" if normalize else "d"

    if num_classes <= 10:
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] < thresh else "black",
            )

    ticks = np.arange(num_classes)
    ax.set(
        xlabel="predicted label",
        ylabel="true label",
        xticks=ticks,
        yticks=ticks,
        xticklabels=classes if num_classes <= 20 else "",
        yticklabels=classes if num_classes <= 20 else "",
    )
    fig.colorbar(im, ax=ax, format="%" + fmt)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath)
    if show:
        plt.show(fig)
    plt.close(fig)
