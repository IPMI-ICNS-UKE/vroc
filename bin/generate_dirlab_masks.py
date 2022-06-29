import os
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from vroc.models import UNet
from vroc.segmentation import LungSegmenter2D


def generate_dirlab_copd_masks(root_path):
    filelist = sorted(glob(os.path.join(root_path, "**/**/phase*.mha")))

    for file in filelist:
        img = sitk.ReadImage(file)
        img_arr = sitk.GetArrayFromImage(img)
        img_arr = np.swapaxes(img_arr, 0, 2)
        out_arr = lung_segmenter.segment(img_arr)
        out_arr = np.swapaxes(out_arr, 0, 2)
        out_arr = out_arr.astype(np.uint8)
        out = sitk.GetImageFromArray(out_arr)
        out.CopyInformation(img)
        sitk.WriteImage(
            out,
            os.path.join(
                root_path,
                os.path.split(os.path.dirname(os.path.dirname(file)))[1],
                "segmentation",
                "mask_" + file[-5:],
            ),
        )


lung_segmenter = LungSegmenter2D(
    model=UNet().to("cuda"),
    iter_axis=0,
    state_filepath=Path(
        "/home/tsentker/Documents/projects/ebolagnul/lung_segmenter.pth"
    ),
)
generate_dirlab_copd_masks(root_path="/home/tsentker/data/copd_dirlab2022/data/")
