from math import ceil
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import map_coordinates


def read_landmarks(filepath):
    with open(filepath) as f:
        lines = [tuple(map(float, line.rstrip().split("\t"))) for line in f]
    return np.array(lines)


def compute_tre(fix_lms, mov_lms, disp, spacing_mov=None, snap_to_voxel=False):
    if not spacing_mov:
        spacing_mov = np.repeat(1, disp.shape[0])
    fix_lms_disp_x = map_coordinates(disp[0, :, :, :], fix_lms.transpose())
    fix_lms_disp_y = map_coordinates(disp[1, :, :, :], fix_lms.transpose())
    fix_lms_disp_z = map_coordinates(disp[2, :, :, :], fix_lms.transpose())
    fix_lms_disp = np.array(
        (fix_lms_disp_x, fix_lms_disp_y, fix_lms_disp_z)
    ).transpose()

    fix_lms_warped = fix_lms + fix_lms_disp
    if snap_to_voxel:
        fix_lms_warped = np.round(fix_lms_warped)

    return np.linalg.norm((fix_lms_warped - mov_lms) * spacing_mov, axis=1)


def compute_tre_sitk(
    fix_lms, mov_lms, transform, ref_img, spacing_mov=None, snap_to_voxel=False
):
    if not spacing_mov:
        spacing_mov = np.repeat(1, ref_img.GetDimensions())
    fix_lms = [ref_img.TransformContinuousIndexToPhysicalPoint(p) for p in fix_lms]
    fix_lms_warped = [np.array(transform.TransformPoint(p)) for p in fix_lms]

    fix_lms_warped = np.array(
        [ref_img.TransformPhysicalPointToContinuousIndex(p) for p in fix_lms_warped]
    )
    if snap_to_voxel:
        fix_lms_warped = np.round(fix_lms_warped)

    return np.linalg.norm((fix_lms_warped - mov_lms) * spacing_mov, axis=1)


def landmark_distance(point_list, reference_point_list):
    return [
        np.linalg.norm(np.array(p) - np.array(p_ref))
        for p, p_ref in zip(point_list, reference_point_list)
    ]


def rescale_range(
    values: np.ndarray, input_range: Tuple, output_range: Tuple, clip: bool = True
):
    in_min, in_max = input_range
    out_min, out_max = output_range
    rescaled = (((values - in_min) * (out_max - out_min)) / (in_max - in_min)) + out_min
    if clip:
        return np.clip(rescaled, out_min, out_max)
    return rescaled


def torch_prepare(image: np.ndarray) -> torch.tensor:
    image = torch.as_tensor(image.copy(), dtype=torch.float32)
    return image[None, :]


def batch_array(array: np.ndarray, batch_size: int = 32):
    n_total = array.shape[0]
    n_batches = ceil(n_total / batch_size)

    for i_batch in range(n_batches):
        yield array[i_batch * batch_size : (i_batch + 1) * batch_size]


def detach_and_squeeze(img, is_vf=False):
    img = img.cpu().detach().numpy()
    img = np.squeeze(img)
    if is_vf:
        img = np.rollaxis(img, 0, img.ndim)
        img = np.swapaxes(img, 0, 2)
        img = sitk.GetImageFromArray(img, isVector=True)
        img = sitk.Cast(img, sitk.sitkVectorFloat64)
    else:
        img = sitk.GetImageFromArray(img)
    return img


def scale_vf(vf, spacing):
    vf = sitk.Compose(
        [sitk.VectorIndexSelectionCast(vf, i) * sp for i, sp in enumerate(spacing)]
    )
    vf = sitk.Cast(vf, sitk.sitkVectorFloat64)
    return vf


def get_bounding_box(mask: torch.Tensor, padding: int = 0):
    def get_axis_bbox(mask, axis: int, padding: int = 0):
        mask_shape = mask.shape
        for i_axis in range(mask.ndim):
            if i_axis == axis:
                continue
            mask = mask.any(dim=i_axis, keepdim=True)

        mask = mask.squeeze()
        mask = torch.where(mask)

        bbox_min = int(mask[0][0])
        bbox_max = int(mask[0][-1])

        bbox_min = max(bbox_min - padding, 0)
        bbox_max = min(bbox_max + padding + 1, mask_shape[axis])

        return slice(bbox_min, bbox_max)

    return tuple(get_axis_bbox(mask, axis=i, padding=padding) for i in range(mask.ndim))
