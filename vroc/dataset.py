from __future__ import annotations

import logging
import random
from functools import cache, partial
from glob import glob
from itertools import combinations
from math import prod
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from vroc.common_types import FloatTuple3D, IntTuple3D, PathLike, SlicingTuple3D
from vroc.decorators import convert
from vroc.hashing import hash_path
from vroc.helper import (
    LazyLoadableList,
    nearest_factor_pow_2,
    rescale_range,
    torch_prepare,
)
from vroc.preprocessing import (
    crop_background,
    resample_image_size,
    resample_image_spacing,
)

logger = logging.getLogger(__name__)


class DatasetMixin:
    @staticmethod
    def load_and_preprocess(filepath: PathLike, is_mask: bool = False) -> sitk.Image:
        filepath = str(filepath)
        dtype = sitk.sitkUInt8 if is_mask else sitk.sitkFloat32
        image = sitk.ReadImage(filepath, dtype)

        return image


class LungCTRegistrationDataset(DatasetMixin):
    def __init__(self):
        super().__init__()

        self._image_pairs = []

    @convert("moving_image", converter=Path)
    @convert("fixed_image", converter=Path)
    @convert("moving_mask", converter=Path)
    @convert("fixed_mask", converter=Path)
    def append_image_pair(
        self,
        moving_image: PathLike,
        fixed_image: PathLike,
        moving_mask: PathLike,
        fixed_mask: PathLike,
    ):
        image_pair = {
            "moving_image": moving_image,
            "fixed_image": fixed_image,
            "moving_mask": moving_mask,
            "fixed_mask": fixed_mask,
        }
        self._image_pairs.append(image_pair)


class LungCTSegmentationDataset(DatasetMixin, IterableDataset):
    def __init__(
        self,
        image_filepaths: List[PathLike],
        mask_filepaths: List[PathLike],
        mask_labels: List[Sequence[int | None]] | None = None,
        patch_shape: IntTuple3D | None = None,
        image_spacing_range: Tuple | None = None,
        random_rotation: bool = True,
        patches_per_image: int | float = 1,
        center_crop: bool = False,
    ):
        self.images = LazyLoadableList(
            image_filepaths, loader=LungCTSegmentationDataset.load_and_preprocess
        )
        self.masks = LazyLoadableList(
            mask_filepaths,
            loader=partial(LungCTSegmentationDataset.load_and_preprocess, is_mask=True),
        )

        self.mask_labels = mask_labels or [None] * len(self.masks)

        self.patch_shape = patch_shape
        self.image_spacing_range = image_spacing_range
        self.random_rotation = random_rotation
        self.patches_per_image = patches_per_image
        self.center_crop = center_crop

        if (
            self.center_crop
            and not isinstance(self.patches_per_image, int)
            and self.patches_per_image != 1
        ):
            raise ValueError("Center crop implies 1 patch per image")

    @staticmethod
    def _resample_image_spacing(
        image: sitk.Image, mask: sitk.Image, image_spacing: FloatTuple3D
    ) -> Tuple[sitk.Image, sitk.Image]:
        image = resample_image_spacing(
            image, new_spacing=image_spacing, default_voxel_value=-1000
        )
        mask = resample_image_spacing(
            mask,
            new_spacing=image_spacing,
            resampler=sitk.sitkNearestNeighbor,
            default_voxel_value=0,
        )

        return image, mask

    @staticmethod
    def sample_random_patch_3d(
        patch_shape: IntTuple3D, image_shape: IntTuple3D
    ) -> SlicingTuple3D:
        if len(patch_shape) != len(image_shape) != 3:
            raise ValueError("Please pass 3D shapes")
        upper_left_index = tuple(
            random.randint(0, s - ps) for (s, ps) in zip(image_shape, patch_shape)
        )

        return tuple(
            slice(ul, ul + ps) for (ul, ps) in zip(upper_left_index, patch_shape)
        )

    @staticmethod
    def random_rotate_image_and_mask(
        image: np.ndarray,
        mask: np.ndarray | None = None,
        spacing: Tuple[int, ...] | None = None,
    ):
        rotation_plane = random.choice(list(combinations(range(image.ndim), 2)))
        n_rotations = random.randint(0, 3)

        image = np.rot90(image, k=n_rotations, axes=rotation_plane)
        if mask is not None:
            mask = np.rot90(mask, k=n_rotations, axes=rotation_plane)

        if spacing:
            spacing = list(spacing)
            if n_rotations % 2:
                spacing[rotation_plane[0]], spacing[rotation_plane[1]] = (
                    spacing[rotation_plane[1]],
                    spacing[rotation_plane[0]],
                )
            spacing = tuple(spacing)

        return image, mask, spacing

    @staticmethod
    def crop_or_pad(
        image: np.ndarray,
        mask: np.ndarray | None,
        target_shape: IntTuple3D,
        image_pad_value=-1000,
        mask_pad_value=0,
        no_crop: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        for i_axis in range(image.ndim):
            if target_shape[i_axis] is not None:
                if image.shape[i_axis] < target_shape[i_axis]:
                    # pad
                    padding = target_shape[i_axis] - image.shape[i_axis]
                    padding_left = padding // 2
                    padding_right = padding - padding_left

                    pad_width = [(0, 0)] * image.ndim
                    pad_width[i_axis] = (padding_left, padding_right)
                    image = np.pad(
                        image,
                        pad_width,
                        mode="constant",
                        constant_values=image_pad_value,
                    )
                    if mask is not None:
                        mask = np.pad(
                            mask,
                            pad_width,
                            mode="constant",
                            constant_values=mask_pad_value,
                        )

                elif not no_crop and image.shape[i_axis] > target_shape[i_axis]:
                    # crop
                    cropping = image.shape[i_axis] - target_shape[i_axis]
                    cropping_left = cropping // 2
                    cropping_right = cropping - cropping_left

                    cropping_slicing = [
                        slice(None, None),
                    ] * image.ndim
                    cropping_slicing[i_axis] = slice(cropping_left, -cropping_right)
                    image = image[tuple(cropping_slicing)]
                    mask = mask[tuple(cropping_slicing)]

        return image, mask

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            self.images.items = self.images.items[worker_id::num_workers]
            self.masks.items = self.masks.items[worker_id::num_workers]
            self.mask_labels = self.mask_labels[worker_id::num_workers]

        for image, image_filepath, mask, mask_filepath, mask_labels in zip(
            self.images,
            self.images.items,
            self.masks,
            self.masks.items,
            self.mask_labels,
        ):

            if self.image_spacing_range is not None:
                # resample to random image spacing
                image_spacing = tuple(
                    float(np.random.uniform(*spacing_range))
                    for spacing_range in self.image_spacing_range
                )
                image, mask = LungCTSegmentationDataset._resample_image_spacing(
                    image, mask, image_spacing=image_spacing
                )

            else:
                image_spacing = image.GetSpacing()

            image_arr = sitk.GetArrayFromImage(image)
            mask_arr = sitk.GetArrayFromImage(mask)

            if (
                isinstance(self.patches_per_image, float)
                and 0.0 < self.patches_per_image <= 1.0
            ):
                # interpret as fraction of image volume
                image_volume = prod(image_arr.shape)
                patch_volume_volume = prod(self.patch_shape)

                patches_per_image = round(
                    (image_volume / patch_volume_volume) * self.patches_per_image
                )

                # at least 1 patch
                patches_per_image = max(patches_per_image, 1)
            else:
                patches_per_image = self.patches_per_image

            if mask_labels is not None:
                mask_arr = np.isin(mask_arr, mask_labels).astype(np.uint8)

            image_arr, mask_arr = image_arr.swapaxes(0, 2), mask_arr.swapaxes(0, 2)

            if image_arr.shape != mask_arr.shape:
                raise RuntimeError("Shape mismatch")

            if self.random_rotation:
                (
                    image_arr,
                    mask_arr,
                    image_spacing,
                ) = LungCTSegmentationDataset.random_rotate_image_and_mask(
                    image_arr, mask=mask_arr, spacing=image_spacing
                )

            if not self.patch_shape:
                # no patching, feed full image: find nearest pow 2 shape
                self.patch_shape = tuple(
                    nearest_factor_pow_2(s) for s in image_arr.shape
                )

            # pad if (rotated) image shape < patch shape
            # also performs center cropping if specified
            image_arr, mask_arr = LungCTSegmentationDataset.crop_or_pad(
                image=image_arr,
                mask=mask_arr,
                target_shape=self.patch_shape,
                no_crop=not self.center_crop,
            )

            for i_patch in range(patches_per_image):
                patch_slicing = LungCTSegmentationDataset.sample_random_patch_3d(
                    patch_shape=self.patch_shape, image_shape=image_arr.shape
                )

                # copy for PyTorch (negative strides are not currently supported)
                image_arr_patch = image_arr[patch_slicing].astype(np.float32, order="C")
                mask_arr_patch = mask_arr[patch_slicing].astype(np.float32, order="C")

                image_arr_patch = rescale_range(
                    image_arr_patch,
                    input_range=(-1024, 3071),
                    output_range=(0, 1),
                    clip=True,
                )

                data = {
                    "id": hash_path(image_filepath)[:7],
                    "image": image_arr_patch[np.newaxis],
                    "mask": mask_arr_patch[np.newaxis],
                    "image_spacing": image_spacing,
                    "full_image_shape": image_arr.shape,
                    "i_patch": i_patch,
                    "n_patches": patches_per_image,
                    "patch_slicing": patch_slicing,
                }

                yield data


class NLSTDataset(DatasetMixin, Dataset):
    def __init__(
        self,
        root_dir: PathLike,
        i_worker: Optional[int] = None,
        n_worker: Optional[int] = None,
    ):
        self.filepaths = self.fetch_filepaths(root_dir)
        self.i_worker = i_worker
        self.n_worker = n_worker

        if i_worker is not None and n_worker is not None:
            self.filepaths = self.filepaths[self.i_worker :: self.n_worker]

    @staticmethod
    @convert("root_dir", Path)
    def fetch_filepaths(
        root_dir: PathLike,
        image_folder: str = "imagesTr",
        mask_folder: str = "masksTr",
        keypoints_folder: str = "keypointsTr",
    ):
        root_dir: Path

        image_path = root_dir / image_folder
        mask_path = root_dir / mask_folder
        keypoints_path = root_dir / keypoints_folder

        fixed_images = sorted(image_path.glob("*0000.nii.gz"))
        moving_images = sorted(image_path.glob("*0001.nii.gz"))
        fixed_masks = sorted(mask_path.glob("*0000.nii.gz"))
        moving_masks = sorted(mask_path.glob("*0001.nii.gz"))
        fixed_keypoints = sorted(keypoints_path.glob("*0000.csv"))
        moving_keypoints = sorted(keypoints_path.glob("*0001.csv"))

        filepath_lists = (
            fixed_images,
            moving_images,
            fixed_masks,
            moving_masks,
            fixed_keypoints,
            moving_keypoints,
        )

        if len(lengths := set(len(l) for l in filepath_lists)) > 1:
            raise RuntimeError("File mismatch")
        length = lengths.pop()

        return [
            {
                "fixed_image": fixed_images[i],
                "moving_image": moving_images[i],
                "fixed_mask": fixed_masks[i],
                "moving_mask": moving_masks[i],
                "fixed_keypoints": fixed_keypoints[i],
                "moving_keypoints": moving_keypoints[i],
            }
            for i in range(length)
        ]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, item):
        fixed_image = NLSTDataset.load_and_preprocess(
            self.filepaths[item]["fixed_image"]
        )
        moving_image = NLSTDataset.load_and_preprocess(
            self.filepaths[item]["moving_image"]
        )
        fixed_mask = NLSTDataset.load_and_preprocess(
            self.filepaths[item]["fixed_mask"], is_mask=True
        )
        moving_mask = NLSTDataset.load_and_preprocess(
            self.filepaths[item]["moving_mask"], is_mask=True
        )

        spacing = torch.tensor(fixed_image.GetSpacing())

        # moving_image = sitk.HistogramMatching(
        #     moving_image,
        #     fixed_image,
        #     numberOfHistogramLevels=1024,
        #     numberOfMatchPoints=7,
        # )

        fixed_image = sitk.GetArrayFromImage(fixed_image)
        moving_image = sitk.GetArrayFromImage(moving_image)
        fixed_mask = sitk.GetArrayFromImage(fixed_mask)
        moving_mask = sitk.GetArrayFromImage(moving_mask)

        patch_shape = torch.tensor(fixed_image.shape)

        fixed_image = torch_prepare(fixed_image)
        moving_image = torch_prepare(moving_image)
        fixed_mask = torch_prepare(fixed_mask)
        moving_mask = torch_prepare(moving_mask)

        return {
            "id": self.filepaths[item]["fixed_image"].name[:9],
            "fixed_image": fixed_image,
            "moving_image": moving_image,
            "fixed_mask": fixed_mask,
            "moving_mask": moving_mask,
            "patch_shape": patch_shape,
            "spacing": spacing,
        }


class AutoencoderDataset(DatasetMixin, Dataset):
    def __init__(self, filepaths: List[PathLike]):
        self.filepaths = filepaths

    @staticmethod
    def load_and_preprocess(filepath: PathLike, is_mask: bool = False) -> np.ndarray:
        filepath = str(filepath)
        dtype = sitk.sitkUInt8 if is_mask else sitk.sitkFloat32
        image = sitk.ReadImage(filepath, dtype)
        image = crop_background(image)
        image = resample_image_size(image, new_size=(128, 128, 128))
        return sitk.GetArrayFromImage(image)

    @staticmethod
    @convert("root_dirs", lambda paths: [Path(p) for p in paths])
    def fetch_filepaths(root_dirs: List[PathLike]):
        root_dirs: List[Path]

        filepaths = []
        allowed_extensions = [".nii.gz", ".mhd", ".mha"]
        for root_dir in root_dirs:
            for ext in allowed_extensions:
                filepaths.extend(
                    sorted(
                        Path(p) for p in glob(str(Path.joinpath(root_dir, "*" + ext)))
                    )
                )

        return filepaths

    def __len__(self):
        return len(self.filepaths)

    @cache
    def __getitem__(self, item):
        image_path = self.filepaths[item]
        image = self.load_and_preprocess(image_path)
        image = rescale_range(image, input_range=(-1024, 3071), output_range=(0, 1))

        image = torch_prepare(image)
        return image, str(image_path)
