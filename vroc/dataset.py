from pathlib import Path

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from vroc.common_types import PathLike
from vroc.decorators import convert
from vroc.helper import load_and_preprocess, rescale_range, torch_prepare
from vroc.preprocessing import crop_background, resample_image_size


class NLSTDataset(Dataset):
    def __init__(self, root_dir: PathLike):
        self.filepaths = self.fetch_filepaths(root_dir)

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

    @staticmethod
    def load_and_preprocess(filepath: PathLike, is_mask: bool = False) -> sitk.Image:
        filepath = str(filepath)
        dtype = sitk.sitkUInt8 if is_mask else sitk.sitkFloat32
        image = sitk.ReadImage(filepath, dtype)

        return image

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

        moving_image = sitk.HistogramMatching(
            moving_image,
            fixed_image,
            numberOfHistogramLevels=1024,
            numberOfMatchPoints=7,
        )
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


class AutoEncoderDataset(VrocDataset):
    def __getitem__(self, item):
        image_path = self.dir_list[item]
        image = load_and_preprocess(image_path)
        image = crop_background(image)
        image = resample_image_size(image, new_size=(128, 128, 128))
        image = sitk.GetArrayFromImage(image)
        image = rescale_range(image, input_range=(-1024, 3071), output_range=(0, 1))
        image = torch_prepare(image)
        return image
