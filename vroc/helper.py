from __future__ import annotations

import threading
from collections import MutableSequence
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
from torch.utils.data import default_collate

from vroc.common_types import FloatTuple3D, Function, IntTuple3D, PathLike


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


def read_landmarks(filepath: PathLike, sep: str = ",") -> np.ndarray:
    with open(filepath, "rt") as f:
        lines = [tuple(map(float, line.strip().split(sep))) for line in f]
    return np.array(lines, dtype=np.float32)


def compute_tre_numpy(
    moving_landmarks: np.ndarray,
    fixed_landmarks: np.ndarray,
    vector_field: np.ndarray | None = None,
    image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
    snap_to_voxel: bool = False,
) -> np.ndarray:
    if vector_field is not None:
        displacement_x = map_coordinates(vector_field[0], fixed_landmarks.transpose())
        displacement_y = map_coordinates(vector_field[1], fixed_landmarks.transpose())
        displacement_z = map_coordinates(vector_field[2], fixed_landmarks.transpose())
        displacement = np.array(
            (displacement_x, displacement_y, displacement_z)
        ).transpose()
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


def remove_suffixes(path: Path) -> Path:
    while path != (without_suffix := path.with_suffix("")):
        path = without_suffix

    return path


def nearest_factor_pow_2(
    value: int, factors: Tuple[int, ...] = (2, 3, 5, 6, 7, 9)
) -> int:
    factors = np.array(factors)
    exponents = np.ceil(np.log2(value / factors))
    nearest = factors * 2**exponents - value

    nearest_factor = factors[np.argmin(nearest)]
    nearest_exponent = exponents[np.argmin(nearest)]
    return int(nearest_factor * 2**nearest_exponent)


def merge_segmentation_labels(
    segmentation: sitk.Image, labels: Sequence[int]
) -> sitk.Image:
    merged = sitk.Image(segmentation.GetSize(), sitk.sitkUInt8)
    merged.CopyInformation(segmentation)

    for label in labels:
        merged = merged | (segmentation == label)

    return merged


def dict_collate(batch, noop_keys: Sequence[Any]) -> dict:
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


def compose_vector_fields(
    vector_field_1: torch.Tensor,
    vector_field_2: torch.Tensor,
    spatial_transformer: "SpatialTransformer" | None = None,
) -> torch.Tensor:
    if (n_dimensions := vector_field_1.ndim) != vector_field_2.ndim != 5:
        raise NotImplementedError(
            "Currently only imlemented for 3D images, i.e. 5D input tensors"
        )

    if (shape := vector_field_1.shape) != vector_field_2.shape:
        raise RuntimeError(
            f"Shape mismatch between vector fields: "
            f"{vector_field_1.shape} vs. {vector_field_2.shape}"
        )

    n_spatial_dimensions = n_dimensions.ndim - 2

    if not spatial_transformer:
        spatial_transformer = SpatialTransformer(
            shape=shape[-n_spatial_dimensions:], mode="bilinear"
        )

    return vector_field_2 + spatial_transformer(vector_field_1, vector_field_2)
