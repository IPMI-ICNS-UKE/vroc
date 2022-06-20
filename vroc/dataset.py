import os

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from vroc.helper import load_and_preprocess, torch_prepare


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
