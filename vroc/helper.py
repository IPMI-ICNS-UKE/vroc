from typing import Tuple

import SimpleITK as sitk
import numpy as np
import matplotlib

matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import ceil


def read_landmarks(filepath):
    with open(filepath) as f:
        lines = [tuple(map(float, line.rstrip().split("\t"))) for line in f]
    return lines


def transform_landmarks_and_flip_z(point_list, reference_image):
    # point_list = [(p[0], p[1], reference_image.GetSize()[2] - p[2]) for p in point_list]
    return [
        reference_image.TransformContinuousIndexToPhysicalPoint(
            (p[0], p[1], reference_image.GetSize()[2] - p[2])
        )
        for p in point_list
    ]


def target_registration_errors_snapped(
    tx, point_list, reference_point_list, reference_image, world=True
):
    """
    Distances between points transformed by the given transformation and their
    location in another coordinate system. When the points are only used to evaluate
    registration accuracy (not used in the registration) this is the target registration
    error (TRE).
    """
    TRE = []
    for p, p_ref in zip(point_list, reference_point_list):
        t_p = np.array(tx.TransformPoint(p))
        t_p_idx = np.round(reference_image.TransformPhysicalPointToContinuousIndex(t_p))
        r_p = np.array(p_ref)
        if world:
            t_p = reference_image.TransformContinuousIndexToPhysicalPoint(t_p_idx)
        else:
            r_p = reference_image.TransformPhysicalPointToContinuousIndex(r_p)
            t_p = t_p_idx
        TRE.append(np.linalg.norm(t_p - r_p))
    return TRE


def target_registration_errors(tx, point_list, reference_point_list):
    """
    Distances between points transformed by the given transformation and their
    location in another coordinate system. When the points are only used to evaluate
    registration accuracy (not used in the registration) this is the target registration
    error (TRE).
    """
    return [np.linalg.norm(np.array(tx.TransformPoint(p)) - np.array(p_ref))
            for p, p_ref in zip(point_list, reference_point_list)]


def landmark_distance(point_list, reference_point_list):
    return [
        np.linalg.norm(np.array(p) - np.array(p_ref))
        for p, p_ref in zip(point_list, reference_point_list)
    ]


def load_and_preprocess(filepath):
    filepath = str(filepath)
    image = sitk.ReadImage(filepath, sitk.sitkFloat32)
    return image


def plot_TRE_landmarks(tx, point_list, reference_point_list):
    transformed_point_list = [tx.TransformPoint(p) for p in point_list]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    orig = ax.scatter(
        list(np.array(reference_point_list).T)[0],
        list(np.array(reference_point_list).T)[1],
        list(np.array(reference_point_list).T)[2],
        marker="o",
        color="blue",
        label="Original points",
    )
    transformed = ax.scatter(
        list(np.array(transformed_point_list).T)[0],
        list(np.array(transformed_point_list).T)[1],
        list(np.array(transformed_point_list).T)[2],
        marker="^",
        color="red",
        label="Transformed points",
    )
    plt.legend(loc=(0.0, 1.0))


def rescale_range(
    values: np.ndarray, input_range: Tuple, output_range: Tuple, clip: bool = True
):
    in_min, in_max = input_range
    out_min, out_max = output_range
    rescaled = (((values - in_min) * (out_max - out_min)) / (in_max - in_min)) + out_min
    if clip:
        return np.clip(rescaled, out_min, out_max)
    return rescaled


def batch_array(array: np.ndarray, batch_size: int = 32):
    n_total = array.shape[0]
    n_batches = ceil(n_total / batch_size)

    for i_batch in range(n_batches):
        yield array[i_batch * batch_size : (i_batch + 1) * batch_size]
