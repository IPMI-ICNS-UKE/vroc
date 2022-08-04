from pathlib import Path

import numpy as np
import SimpleITK as sitk
from ipmi.registration.ct.landmarks import extract_lung_landmarks

moving_images = sorted(
    Path("/datalake/learn2reg/NLST_Validation/imagesTr").glob("*_0001.nii.gz")
)
fixed_images = sorted(
    Path("/datalake/learn2reg/NLST_Validation/imagesTr").glob("*_0000.nii.gz")
)
fixed_masks = sorted(
    Path("/datalake/learn2reg/NLST_Validation/masksTr").glob("*_0000.nii.gz")
)

keypoints_folder = Path("/datalake/learn2reg/NLST_Validation/keypointsTr")
keypoints_folder.mkdir(exist_ok=True)

for moving_image_filepath, fixed_image_filepath, fixed_mask_filepath in zip(
    moving_images, fixed_images, fixed_masks
):
    moving_image = sitk.ReadImage(str(moving_image_filepath))
    fixed_image = sitk.ReadImage(str(fixed_image_filepath))
    fixed_mask = sitk.ReadImage(str(fixed_mask_filepath))

    landmarks = extract_lung_landmarks(
        moving_image=moving_image, fixed_image=fixed_image, fixed_mask=fixed_mask
    )

    np.savetxt(
        keypoints_folder / (fixed_image_filepath.name.split(".")[0] + ".csv"),
        [l["fixed_image"] for l in landmarks["landmarks"]],
        delimiter=",",
        fmt="%.3f",
    )
    np.savetxt(
        keypoints_folder / (moving_image_filepath.name.split(".")[0] + ".csv"),
        [l["moving_image"] for l in landmarks["landmarks"]],
        delimiter=",",
        fmt="%.3f",
    )
