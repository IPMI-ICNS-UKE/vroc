import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path

from vroc.segmentation import LungSegmenter2D
from vroc.models import UNet

dirlab_case = 8
phase = (0, 5)
DIRLAB_PATH = f"/home/tsentker/data/dirlab2022/data/Case{dirlab_case:02d}Pack"
data_path = os.path.join(DIRLAB_PATH, "Images")
out_path = os.path.join(DIRLAB_PATH, "segmentation")

lung_segmenter = LungSegmenter2D(
    model=UNet().to("cuda"),
    iter_axis=0,
    state_filepath=Path(
        "/home/tsentker/Documents/projects/ebolagnul/lung_segmenter.pth"
    ),
)

for p in phase:
    img = sitk.ReadImage(os.path.join(data_path, f"phase_{p}.mha"))
    img_arr = sitk.GetArrayFromImage(img)
    # img_arr = np.swapaxes(img_arr, 0, 2)
    out_mask_arr = lung_segmenter.segment(image=img_arr)
    # out_mask_arr = np.swapaxes(out_mask_arr, 0, 2)
    out_mask_arr = out_mask_arr.astype(int)
    out_mask = sitk.GetImageFromArray(out_mask_arr)
    out_mask = sitk.Cast(out_mask, sitk.sitkUInt8)
    out_mask.CopyInformation(img)
    out_name = f"ebolagnul_mask_{p}.mha"
    sitk.WriteImage(out_mask, os.path.join(out_path, out_name))
