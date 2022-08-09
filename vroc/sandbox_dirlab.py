import logging
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage.morphology import binary_dilation

from vroc.helper import compute_tre_numpy, compute_tre_sitk, read_landmarks
from vroc.logger import LogFormatter
from vroc.registration import VrocRegistration

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(LogFormatter())
logging.basicConfig(handlers=[handler])

logging.getLogger(__name__).setLevel(logging.DEBUG)
logging.getLogger("vroc").setLevel(logging.DEBUG)
logging.getLogger("vroc.models.VarReg3d").setLevel(logging.INFO)

ROOT_DIR = (
    Path("/home/tsentker/data"),
    Path("/datalake/learn2reg"),
)
ROOT_DIR = next(p for p in ROOT_DIR if p.exists())
FOLDER = "dirlab2022"

device = "cuda:0"

OUTPUT_FOLDER = Path(f"{ROOT_DIR}/{FOLDER}/predictions")


def load(
    moving_image_filepath,
    fixed_image_filepath,
    moving_mask_filepath,
    fixed_mask_filepath,
):
    moving_image = sitk.ReadImage(moving_image_filepath, sitk.sitkFloat32)
    fixed_image = sitk.ReadImage(fixed_image_filepath, sitk.sitkFloat32)
    moving_mask = sitk.ReadImage(moving_mask_filepath)
    fixed_mask = sitk.ReadImage(fixed_mask_filepath)

    reference_image = fixed_image

    image_spacing = fixed_image.GetSpacing()[::-1]

    moving_image = sitk.GetArrayFromImage(moving_image)
    fixed_image = sitk.GetArrayFromImage(fixed_image)
    moving_mask = sitk.GetArrayFromImage(moving_mask)
    fixed_mask = sitk.GetArrayFromImage(fixed_mask)

    moving_image = np.swapaxes(moving_image, 0, 2)
    fixed_image = np.swapaxes(fixed_image, 0, 2)
    moving_mask = np.swapaxes(moving_mask, 0, 2)
    fixed_mask = np.swapaxes(fixed_mask, 0, 2)

    return (
        moving_image,
        fixed_image,
        moving_mask,
        fixed_mask,
        image_spacing,
        reference_image,
    )


params = {
    "iterations": 800,
    "tau": 2.25,
    "sigma_x": 2,
    "sigma_y": 2,
    "sigma_z": 2,
    "n_levels": 3,
}

registration = VrocRegistration(
    roi_segmenter=None,
    feature_extractor=None,
    parameter_guesser=None,
    default_parameters=params,
    debug=False,
    device="cuda:0",
)

tres_before = []
tres_after = []
t_start = time.time()
for case in range(1, 11):
    fixed_landmarks = read_landmarks(
        f"{ROOT_DIR}/{FOLDER}/data/Case{case:02d}Pack/extremePhases/landmarks_0.txt",
        sep="\t",
    )
    moving_landmarks = read_landmarks(
        f"{ROOT_DIR}/{FOLDER}/data/Case{case:02d}Pack/extremePhases/landmarks_5.txt",
        sep="\t",
    )

    (
        moving_image,
        fixed_image,
        moving_mask,
        fixed_mask,
        image_spacing,
        reference_image,
    ) = load(
        moving_image_filepath=f"{ROOT_DIR}/{FOLDER}/data/Case{case:02d}Pack/Images/phase_5.mha",
        fixed_image_filepath=f"{ROOT_DIR}/{FOLDER}/data/Case{case:02d}Pack/Images/phase_0.mha",
        moving_mask_filepath=f"{ROOT_DIR}/{FOLDER}/data/Case{case:02d}Pack/segmentation/mask_5.mha",
        fixed_mask_filepath=f"{ROOT_DIR}/{FOLDER}/data/Case{case:02d}Pack/segmentation/mask_0.mha",
    )

    moving_mask = binary_dilation(moving_mask.astype(np.uint8), iterations=1).astype(
        np.uint8
    )
    fixed_mask = binary_dilation(fixed_mask.astype(np.uint8), iterations=1).astype(
        np.uint8
    )

    union_mask = moving_mask | fixed_mask

    warped_image, vector_field = registration.register(
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_mask=union_mask,
        fixed_mask=union_mask,
        register_affine=True,
        valid_value_range=(-1024, 3071),
        early_stopping_delta=1e-6,
        early_stopping_window=250,
    )

    vf = torch.as_tensor(vector_field[np.newaxis], device=device)
    ml = torch.as_tensor(moving_landmarks[np.newaxis], device=device)
    fl = torch.as_tensor(fixed_landmarks[np.newaxis], device=device)

    tre_before = compute_tre_numpy(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=None,
        image_spacing=reference_image.GetSpacing(),
    )
    tre_after = compute_tre_numpy(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=vector_field,
        image_spacing=reference_image.GetSpacing(),
        snap_to_voxel=True,
    )

    print(
        f"dirlab_{case:02d}: "
        f"tre_before={np.mean(tre_before):.2f}, "
        f"tre_after={np.mean(tre_after):.2f}, "
    )
    tres_before.append(np.mean(tre_before))
    tres_after.append(np.mean(tre_after))

print(f"before: mean TRE={np.mean(tres_before)}, std TRE={np.std(tres_before)}")
print(f"after: mean TRE={np.mean(tres_after)}, std TRE={np.std(tres_after)}")

print(f"run took {time.time() - t_start}")
