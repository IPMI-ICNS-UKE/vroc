import os

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from vroc.helper import load_and_preprocess, rescale_range, torch_prepare
from vroc.preprocessing import crop_background, resample_image_size


class VrocDataset(Dataset):
    def __init__(self, dir_list):
        self.dir_list = dir_list

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, item):
        fixed_path = self.dir_list[item]["fixed"]
        mask_path = self.dir_list[item]["mask"]
        moving_path = self.dir_list[item]["moving"]

        fixed_image = load_and_preprocess(fixed_path)
        spacing = torch.tensor(fixed_image.GetSpacing())
        moving_image = load_and_preprocess(moving_path)
        mask = sitk.ReadImage(mask_path)
        moving_image = sitk.HistogramMatching(
            moving_image,
            fixed_image,
            numberOfHistogramLevels=1024,
            numberOfMatchPoints=7,
        )
        fixed_image = sitk.GetArrayFromImage(fixed_image)
        moving_image = sitk.GetArrayFromImage(moving_image)
        mask = sitk.GetArrayFromImage(mask)

        patch_shape = torch.tensor(fixed_image.shape)

        fixed_image = torch_prepare(fixed_image)
        moving_image = torch_prepare(moving_image)
        mask = torch_prepare(mask)

        return {
            "fixed_image": fixed_image,
            "mask": mask,
            "moving_image": moving_image,
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
