from __future__ import annotations

import logging
import threading
from collections.abc import MutableSequence
from math import ceil
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

from vroc.common_types import (
    ArrayOrTensor,
    FloatTuple3D,
    Function,
    IntTuple3D,
    PathLike,
)

logger = logging.getLogger(__name__)


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        """This function transforms generator into a background-thead
        generator.

        :param generator: generator or genexp or any
        It can be used with any minibatch generator.
        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.
        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.
        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.
        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!
        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.
        """
        threading.Thread.__init__(self)
        self.queue = Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()
        self.exhausted = False

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        if self.exhausted:
            raise StopIteration
        else:
            next_item = self.queue.get()
            if next_item is None:
                raise StopIteration
            return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class LazyLoadableList(MutableSequence):
    def __init__(self, sequence, loader: Function | None = None):
        super().__init__()

        self._loader = loader
        self.items = list(sequence)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        if self._loader:
            item = self._loader(item)

        return item

    def __setitem__(self, index, value):
        self.items[index] = value

    def __delitem__(self, index):
        del self.items[index]

    def insert(self, index, value):
        self.items.insert(index, value)

    def append(self, value):
        self.insert(len(self) + 1, value)

    def __repr__(self):
        return repr(self.items)


def read_landmarks(filepath: PathLike, sep: str | None = None) -> np.ndarray:
    possible_seps = (" ", "\t", ",")
    with open(filepath, "rt") as f:
        lines = [line for line in f]

    if not sep:
        # guesstimate separator
        for sep in possible_seps:
            if sep in lines[0]:
                break
        else:
            raise RuntimeError(
                "Could not guesstimate separator. Please specify separator."
            )

    lines = [tuple(map(float, line.strip().split(sep))) for line in lines]
    return np.array(lines, dtype=np.float32)


def compute_tre_numpy(
    moving_landmarks: np.ndarray,
    fixed_landmarks: np.ndarray,
    vector_field: np.ndarray | None = None,
    image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
    snap_to_voxel: bool = False,
    fixed_bounding_box: Tuple[slice, slice, slice] | None = None,
) -> np.ndarray | None:
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

    return np.linalg.norm(
        (fixed_landmarks_warped - moving_landmarks) * image_spacing, axis=1
    )


def compute_dice(moving_mask, fixed_mask, moving_warped_mask, labels):
    dice = 0
    count = 0
    for i in labels:
        if ((fixed_mask == i).sum() == 0) or ((moving_mask == i).sum() == 0):
            continue
        dice += compute_dice_coefficient((fixed_mask == i), (moving_warped_mask == i))
        count += 1
    dice /= count
    return dice


def compute_dice_coefficient(mask_gt, mask_pred):
    """Computes soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.
    Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return 0

    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def compute_tre_sitk(
    fix_lms,
    mov_lms,
    transform=None,
    ref_img=None,
    spacing_mov=None,
    snap_to_voxel=False,
):
    if transform and ref_img:
        if not spacing_mov:
            spacing_mov = np.repeat(1, ref_img.GetDimensions())
        fix_lms = [ref_img.TransformContinuousIndexToPhysicalPoint(p) for p in fix_lms]
        fix_lms_warped = [np.array(transform.TransformPoint(p)) for p in fix_lms]

        fix_lms_warped = np.array(
            [ref_img.TransformPhysicalPointToContinuousIndex(p) for p in fix_lms_warped]
        )
    else:
        fix_lms_warped = fix_lms
    if snap_to_voxel:
        fix_lms_warped = np.round(fix_lms_warped)

    return np.linalg.norm((fix_lms_warped - mov_lms) * spacing_mov, axis=1)


def rescale_range(
    values: ArrayOrTensor, input_range: Tuple, output_range: Tuple, clip: bool = True
):
    if input_range and output_range:
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
    labels_shape[dim] = n_classes

    labels_one_hot = torch.zeros(size=labels_shape, dtype=dtype, device=labels.device)
    labels_one_hot.scatter_(dim=dim, index=labels.long(), value=1)

    return labels_one_hot


def torch_prepare(image: np.ndarray) -> torch.tensor:
    image = torch.as_tensor(image.copy(), dtype=torch.float32)
    return image[None]


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
        upper_exponents = upper_exponents[upper_exponents >= min_exponent]
        lower_exponents = lower_exponents[lower_exponents >= min_exponent]

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


def merge_segmentation_labels(
    segmentation: sitk.Image, labels: Sequence[int]
) -> sitk.Image:
    merged = sitk.Image(segmentation.GetSize(), sitk.sitkUInt8)
    merged.CopyInformation(segmentation)

    for label in labels:
        merged = merged | (segmentation == label)

    return merged


def dict_collate(batch, noop_keys: Sequence[Any]) -> dict:
    from torch.utils.data import default_collate

    batch_torch = [
        {key: value for (key, value) in b.items() if key not in noop_keys}
        for b in batch
    ]

    batch_noop = [
        {key: value for (key, value) in b.items() if key in noop_keys} for b in batch
    ]

    batch_torch = default_collate(batch_torch)
    batch_noop = concat_dicts(batch_noop)

    return batch_torch | batch_noop


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


def binary_dilation(mask: torch.Tensor, kernel_size: IntTuple3D) -> torch.Tensor:
    kernel = torch.ones(
        (
            1,
            1,
        )
        + kernel_size,
        dtype=torch.float32,
        device=mask.device,
    )
    dilated = F.conv3d(mask, kernel, padding="same")

    return torch.clip(dilated, 0, 1)


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure if there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


from collections import Sequence


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


if __name__ == "__main__":
    labels = torch.ones((1, 1, 10))
    labels[..., 0:3] = 0
    labels[..., 3:6] = 1
    labels[..., 6:] = 2

    l = to_one_hot(labels, n_classes=3)
