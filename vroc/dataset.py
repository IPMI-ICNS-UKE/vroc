import os
import torch
from torch.utils.data import Dataset
from vroc.helper import load_and_preprocess, torch_prepare
import SimpleITK as sitk


class VrocDataset(Dataset):
    def __init__(self, dir_list):
        self.dir_list = dir_list

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, item):
        fixed_path = self.dir_list[item]["fixed"]
        mask_path = self.dir_list[item]["mask"]
        moving_path = self.dir_list[item]["moving"]

        fixed = load_and_preprocess(fixed_path)
        spacing = torch.tensor(fixed.GetSpacing())
        moving = load_and_preprocess(moving_path)
        mask = sitk.ReadImage(mask_path)
        moving = sitk.HistogramMatching(
            moving, fixed, numberOfHistogramLevels=1024, numberOfMatchPoints=7
        )
        fixed = sitk.GetArrayFromImage(fixed)
        moving = sitk.GetArrayFromImage(moving)
        mask = sitk.GetArrayFromImage(mask)

        patch_shape = torch.tensor(fixed.shape)

        fixed = torch_prepare(fixed)
        moving = torch_prepare(moving)
        mask = torch_prepare(mask)

        return {
            "fixed": fixed,
            "mask": mask,
            "moving": moving,
            "patch_shape": patch_shape,
            "spacing": spacing,
        }
