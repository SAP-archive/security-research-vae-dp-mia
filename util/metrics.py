from typing import List, Union

import numpy as np
from skimage.metrics import structural_similarity as ssim


def calc_mc_accuracy(heuristic: Union[str, int], name: str, results: dict) -> float:
    accuracies = np.array([x[heuristic][name] for x in results])
    return accuracies.mean()


def calc_mc_set_accuracy(heuristic: Union[str, int], name: str, results: dict) -> float:
    accuracies = np.array([x[heuristic][name] for x in results])
    advantages = calc_advantage(accuracies)
    probabilities = np.array(list(map(calc_probability, advantages)))
    mean = probabilities.mean()
    return mean


def calc_probability(advantage: float) -> int:
    if advantage == 0.0:
        return 0.5
    return int(advantage > 0)


def calc_advantage(accuracies):
    advantages = (accuracies - 0.5) * 2
    return advantages


def calc_ssim_for_batch(
    orig: np.array, pred: np.array, data_shape: tuple
) -> List[float]:
    ssim_metric = np.fromiter(
        (
            ssim(im_first, im_second, multichannel=(len(data_shape) > 3))
            for im_first, im_second in zip(orig, pred)
        ),
        dtype=np.float16,
        count=data_shape[0],
    )

    return ssim_metric
