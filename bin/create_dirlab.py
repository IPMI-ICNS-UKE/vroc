import os
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_meta(filepath: Path) -> dict:
    with open(filepath, "rt") as f:
        lines = f.readlines()

    match = re.search("(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", lines[0])
    image_shape = tuple(int(val) for val in match.groups())

    match = re.search(
        "(([0-9]*[.])?[0-9]+)\s*x\s*(([0-9]*[.])?[0-9]+)\s*x\s*(([0-9]*[.])?[0-9]+)",
        lines[1],
    )
    voxel_spacing = tuple(float(val) for val in match.groups()[0::2])

    meta = dict(image_shape=image_shape, voxel_spacing=voxel_spacing)

    return meta


def read_image(filepath: Path) -> sitk.Image:
    folder = filepath.parents[1]
    meta = read_meta(folder / "meta.txt")

    image = np.fromfile(filepath, dtype=np.int16)
    image = image - 1024
    image = np.clip(image, -1024, 3071)
    image = image.reshape(meta["image_shape"][::-1])
    image = np.flip(image, axis=0)

    image = sitk.GetImageFromArray(image)
    image.SetSpacing(meta["voxel_spacing"])

    return image


if __name__ == "__main__":
    DIRLAB_FOLDER = Path("/home/tsentker/data/dirlab2022/")
    DIRLAB_FOLDER_OLD = Path("/home/tsentker/data/dirlab/Segm_asMHD")

    for image_filepath in sorted(DIRLAB_FOLDER.glob("**/*.img")):

        phase = int(re.search("T(\d\d)", str(image_filepath)).group(1)) // 10

        print(phase, image_filepath)
        image = read_image(image_filepath)
        output_filepath = image_filepath.parent / f"phase_{phase}.mha"
        sitk.WriteImage(image, str(output_filepath))

    # for f in sorted(DIRLAB_FOLDER_OLD.glob("**/**/*Lungs.mhd")):
    #     case = int(re.search("Case(\d\d)", str(f)).group(1))
    #     phase = int(re.search("T(\d\d)", str(f)).group(1)) // 10
    #     output_filepath = os.path.join(DIRLAB_FOLDER, 'data', f'Case{case:02d}Pack', 'segmentation', f'mask_{phase}.mha')
    #     segm = sitk.ReadImage(str(f))
    #     segm_arr = sitk.GetArrayFromImage(segm)
    #     segm_arr = np.flip(segm_arr, axis=0)
    #     segm_img = sitk.GetImageFromArray(segm_arr)
    #     segm_img.SetSpacing(segm.GetSpacing())
    #     # segm_img.SetOrigin((0.0, 0.0, 0.0))
    #     segm_img = sitk.Cast(segm_img, sitk.sitkInt8)
    #     sitk.WriteImage(segm_img, str(output_filepath))
    #     print(f, output_filepath)
    #     # break
