import os

from core.dataset import *


def pertub_images(data_container, list_of_epsilon, list_of_datasets):
    m = 64
    b = 1
    data_folder = "/path/to/data/"
    dc = globals()[data_container]

    for dataset in list_of_datasets:
        path_to_orig_data = os.path.join(data_folder, f"{dataset}", f"{dataset}.npz")
        path_to_eps_data_template = os.path.join(
            data_folder, f"{dataset}", f"{dataset}_eps_" + "{}" + f"_m_{m}_b_{b}.npz"
        )
        for eps in list_of_epsilon:
            dc.perturb(
                path_to_orig_data, path_to_eps_data_template.format(eps), eps, m, b
            )


def pertub_timeseries(data_container, list_of_epsilon, list_of_datasets):
    data_folder = "/path/to/data/"
    dc = globals()[data_container]

    for dataset in list_of_datasets:
        path_to_eps_data_template = os.path.join(
            data_folder, f"{dataset}", f"{dataset}_eps_" + "{}.npz"
        )
        for eps in list_of_epsilon:
            dc.perturb(None, path_to_eps_data_template.format(eps), eps)


data_container = "InsertDataContainerName"
list_of_epsilon = ["eps", "value"]
list_of_datasets = ["datasets", "to", "perturb"]

pertub_images(data_container, list_of_epsilon, list_of_datasets)
pertub_timeseries(data_container, list_of_epsilon, list_of_datasets)
