from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Hashable, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import map_coordinates

from vroc.common_types import ArrayOrTensor, FloatTuple3D, PathLike
from vroc.decorators import convert

if TYPE_CHECKING:
    from vroc.registration import RegistrationResult

logger = logging.getLogger(__name__)


def compute_tre_numpy(
    moving_landmarks: np.ndarray,
    fixed_landmarks: np.ndarray,
    vector_field: np.ndarray | None = None,
    image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
    snap_to_voxel: bool = False,
    fixed_bounding_box: Tuple[slice, slice, slice] | None = None,
    axis: int | None = None,
) -> (np.ndarray | None, np.ndarray | None):
    if fixed_bounding_box:
        _, keypoint_mask = mask_keypoints(
            keypoints=fixed_landmarks, bounding_box=fixed_bounding_box
        )
        if not keypoint_mask.any():
            logger.warning("No landmarks inside given fixed bounding box")

        fixed_landmarks = fixed_landmarks[keypoint_mask]
        moving_landmarks = moving_landmarks[keypoint_mask]

    if vector_field is not None:
        # order 1: linear interpolation if vector field at fixed landmarks
        displacement_x = map_coordinates(vector_field[0], fixed_landmarks.T, order=1)
        displacement_y = map_coordinates(vector_field[1], fixed_landmarks.T, order=1)
        displacement_z = map_coordinates(vector_field[2], fixed_landmarks.T, order=1)
        displacement = np.array((displacement_x, displacement_y, displacement_z)).T
        fixed_landmarks_warped = fixed_landmarks + displacement
    else:
        fixed_landmarks_warped = fixed_landmarks

    if snap_to_voxel:
        fixed_landmarks_warped = np.round(fixed_landmarks_warped)

    if axis is not None:
        axis_slicing = np.index_exp[:, axis : axis + 1]
        fixed_landmarks_warped = fixed_landmarks_warped[axis_slicing]
        moving_landmarks = moving_landmarks[axis_slicing]
        image_spacing = image_spacing[axis]

    tre = np.linalg.norm(
        (fixed_landmarks_warped - moving_landmarks) * image_spacing, axis=1
    )
    return tre, fixed_landmarks_warped


def rescale_range(
    values: ArrayOrTensor, input_range: Tuple, output_range: Tuple, clip: bool = True
):
    if input_range and output_range and (input_range != output_range):
        is_tensor = isinstance(values, torch.Tensor)
        in_min, in_max = input_range
        out_min, out_max = output_range
        values = (
            ((values - in_min) * (out_max - out_min)) / (in_max - in_min)
        ) + out_min
        if clip:
            clip_func = torch.clip if is_tensor else np.clip
            values = clip_func(values, out_min, out_max)

    return values


def to_one_hot(
    labels: torch.Tensor,
    n_classes: int | None = None,
    dtype: torch.dtype = torch.float32,
    dim: int = 1,
) -> torch.Tensor:
    labels_shape = list(labels.shape)

    if labels_shape[dim] != 1:
        raise ValueError(
            f"Labels should be single channel, "
            f"got {labels_shape[dim]} channels instead"
        )

    if n_classes is None:
        # guess number of classes
        n_classes = labels.max()
    labels_shape[dim] = int(n_classes)

    labels_one_hot = torch.zeros(size=labels_shape, dtype=dtype, device=labels.device)
    labels_one_hot.scatter_(dim=dim, index=labels.long(), value=1)

    return labels_one_hot


def get_robust_bounding_box_3d(
    image: np.ndarray, bbox_range: Tuple[float, float] = (0.01, 0.99), padding: int = 0
) -> Tuple[slice, slice, slice]:
    x = np.cumsum(image.sum(axis=(1, 2)))
    y = np.cumsum(image.sum(axis=(0, 2)))
    z = np.cumsum(image.sum(axis=(0, 1)))

    x = x / x[-1]
    y = y / y[-1]
    z = z / z[-1]

    x_min, x_max = np.searchsorted(x, bbox_range[0]), np.searchsorted(x, bbox_range[1])
    y_min, y_max = np.searchsorted(y, bbox_range[0]), np.searchsorted(y, bbox_range[1])
    z_min, z_max = np.searchsorted(z, bbox_range[0]), np.searchsorted(z, bbox_range[1])

    x_min, x_max = max(x_min - padding, 0), min(x_max + padding, image.shape[0])
    y_min, y_max = max(y_min - padding, 0), min(y_max + padding, image.shape[1])
    z_min, z_max = max(z_min - padding, 0), min(z_max + padding, image.shape[2])

    return np.index_exp[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]


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


def mask_keypoints(
    keypoints: torch.Tensor | np.ndarray, bounding_box: Tuple[slice, ...]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # keypoints has shape (1, n_keypoints, n_dim)
    # check dimensions
    if len(bounding_box) != keypoints.shape[-1]:
        raise ValueError("Dimension mismatch")
    total_mask = None
    for i_axis, axis_bbox in enumerate(bounding_box):
        axis_mask = (keypoints[..., i_axis] >= axis_bbox.start) & (
            keypoints[..., i_axis] < axis_bbox.stop
        )

        if total_mask is None:
            total_mask = axis_mask
        else:
            total_mask &= axis_mask

    # remove 1 from shape in case of tensor
    if len(total_mask.shape) == 2:
        total_mask = total_mask[0]
        masked_keypoints = keypoints[:, total_mask]
    else:
        masked_keypoints = keypoints[total_mask]

    return masked_keypoints, total_mask


def remove_suffixes(path: Path) -> Path:
    while path != (without_suffix := path.with_suffix("")):
        path = without_suffix

    return path


def nearest_factor_pow_2(
    value: int,
    factors: Tuple[int, ...] = (2, 3, 5, 6, 7, 9),
    min_exponent: int | None = None,
    max_value: int | None = None,
    allow_smaller_value: bool = False,
) -> int:
    factors = np.array(factors)
    upper_exponents = np.ceil(np.log2(value / factors))
    lower_exponents = upper_exponents - 1

    if min_exponent:
        upper_exponents[upper_exponents < min_exponent] = np.inf
        lower_exponents[lower_exponents < min_exponent] = np.inf

    def get_distances(
        factors: Tuple[int, ...], exponents: Tuple[int, ...], max_value: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pow2_values = factors * 2**exponents
        if max_value:
            mask = pow2_values <= max_value
            pow2_values = pow2_values[mask]
            factors = factors[mask]
            exponents = exponents[mask]

        return np.abs(pow2_values - value), factors, exponents

    distances, _factors, _exponents = get_distances(
        factors=factors, exponents=upper_exponents, max_value=max_value
    )
    if len(distances) == 0:
        if allow_smaller_value:
            distances, _factors, _exponents = get_distances(
                factors=factors, exponents=lower_exponents, max_value=max_value
            )
        else:
            raise RuntimeError("Could not find a value")

    if len(distances):
        nearest_factor = _factors[np.argmin(distances)]
        nearest_exponent = _exponents[np.argmin(distances)]
    else:
        # nothing found
        pass

    return int(nearest_factor * 2**nearest_exponent)


def pad_bounding_box_to_pow_2(
    bounding_box: Tuple[slice, ...],
    factors: Tuple[int, ...] = (2, 3, 5, 6, 7, 9),
    reference_shape: Tuple[int, ...] | None = None,
) -> tuple[slice, ...]:
    if any([b.step and b.step > 1 for b in bounding_box]):
        raise NotImplementedError("Only step size of 1 for now")

    n_dim = len(bounding_box)
    bbox_shape = tuple(b.stop - b.start for b in bounding_box)
    if reference_shape:
        print(bounding_box)
        print(bbox_shape, reference_shape)
        padding = tuple(
            nearest_factor_pow_2(
                s, factors=factors, max_value=r, allow_smaller_value=True
            )
            - s
            for s, r in zip(bbox_shape, reference_shape)
        )
    else:
        padding = tuple(
            nearest_factor_pow_2(s, factors=factors) - s for s in bbox_shape
        )

    padded_bbox = []
    for i_axis in range(n_dim):
        padding_left = padding[i_axis] // 2
        padding_right = padding[i_axis] - padding_left

        padded_slice = slice(
            bounding_box[i_axis].start - padding_left,
            bounding_box[i_axis].stop + padding_right,
        )

        if padded_slice.start < 0:
            padded_slice = slice(
                0,
                padded_slice.stop - padded_slice.start,
            )

        padded_bbox.append(padded_slice)
    return tuple(padded_bbox)


def concat_dicts(dicts: Sequence[dict], extend_lists: bool = False):
    concat = {}
    for d in dicts:
        for key, value in d.items():
            try:
                if extend_lists and isinstance(value, list):
                    concat[key].extend(value)
                else:
                    concat[key].append(value)
            except KeyError:
                if extend_lists and isinstance(value, list):
                    concat[key] = value
                else:
                    concat[key] = [value]

    return concat


def convert_dict_values(d: dict, types, converter):
    types = tuple(types)

    if isinstance(d, (list, tuple, set)):
        seq_type: type = type(d)
        return seq_type(
            [convert_dict_values(_d, types=types, converter=converter) for _d in d]
        )

    elif isinstance(d, types):
        return converter(d)

    elif isinstance(d, dict):
        # copy dict so we do not modify the original dict
        d = d.copy()
        for key, _d in d.items():
            d[key] = convert_dict_values(_d, types=types, converter=converter)

        return d

    else:
        return d


def write_vector_field(
    vector_field: np.ndarray,
    output_filepath: PathLike,
):
    vector_field = np.swapaxes(vector_field, 1, 3)
    vector_field = sitk.GetImageFromArray(vector_field, isVector=False)

    sitk.WriteImage(vector_field, str(output_filepath))


@convert("output_folder", converter=Path)
def write_registration_result(
    registration_result: RegistrationResult, output_folder: PathLike
):
    output_folder: Path
    output_folder.mkdir(parents=True, exist_ok=True)

    # write warped image
    warped_image = np.swapaxes(registration_result.warped_moving_image, 0, 2)
    warped_image = sitk.GetImageFromArray(warped_image)

    sitk.WriteImage(warped_image, str(output_folder / "warped_image.nii"))

    write_vector_field(
        vector_field=registration_result.composed_vector_field,
        output_filepath=output_folder / "vector_field.nii",
    )


def get_mode_from_alternation_scheme(
    alternation_scheme: dict[Hashable, int], iteration: int
) -> Hashable:
    total_iterations = sum(alternation_scheme.values())
    residual = iteration % total_iterations
    for mode, mode_iterations in alternation_scheme.items():
        residual -= mode_iterations
        if residual < 0:
            return mode
