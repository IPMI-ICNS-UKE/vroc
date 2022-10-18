import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from matplotlib.animation import FFMpegWriter
from scipy.ndimage.morphology import binary_dilation
from torch.optim import Adam

from vroc.common_types import PathLike
from vroc.feature_extractor import OrientedHistogramFeatureExtrator
from vroc.guesser import ParameterGuesser
from vroc.helper import (
    compute_tre_numpy,
    compute_tre_sitk,
    read_landmarks,
    rescale_range,
)
from vroc.l2r_eval import calculate_l2r_smoothness
from vroc.logger import init_fancy_logging
from vroc.loss import TRELoss, mse_loss, ncc_loss, ngf_loss
from vroc.metrics import jacobian_determinant
from vroc.models import DemonsVectorFieldBooster
from vroc.registration import VrocRegistration

init_fancy_logging()

logger = logging.getLogger(__name__)


logger.setLevel(logging.DEBUG)
logging.getLogger("vroc").setLevel(logging.INFO)
logging.getLogger("vroc.affine").setLevel(logging.DEBUG)
logging.getLogger("vroc.models.VarReg").setLevel(logging.INFO)

ROOT_DIR = (
    Path("/home/tsentker/data/learn2reg"),
    Path("/datalake/learn2reg"),
)
ROOT_DIR = next(p for p in ROOT_DIR if p.exists())
FOLDER = "LungCT"

OUTPUT_FOLDER = Path(f"{ROOT_DIR}/{FOLDER}/predictions/non_boosted")
OUTPUT_FOLDER.mkdir(exist_ok=True)

DEVICE = "cuda:0"


def write_nlst_vector_field(
    vector_field: np.ndarray,
    case: str,
    output_folder: Path,
):
    vector_field = np.swapaxes(vector_field, 1, 3)
    vector_field = sitk.GetImageFromArray(vector_field, isVector=False)

    output_filepath = output_folder / f"disp_{case}_{case}.nii.gz"
    sitk.WriteImage(vector_field, str(output_filepath))

    return str(output_filepath)


def load(
    moving_image_filepath,
    fixed_image_filepath,
    moving_mask_filepath,
    fixed_mask_filepath,
):
    moving_image = sitk.ReadImage(moving_image_filepath)
    fixed_image = sitk.ReadImage(fixed_image_filepath)
    moving_mask = sitk.ReadImage(moving_mask_filepath)
    fixed_mask = sitk.ReadImage(fixed_mask_filepath)

    reference_image = fixed_image

    image_spacing = fixed_image.GetSpacing()[::-1]

    moving_image = sitk.GetArrayFromImage(moving_image)
    fixed_image = sitk.GetArrayFromImage(fixed_image)
    moving_mask = sitk.GetArrayFromImage(moving_mask).astype(bool)
    fixed_mask = sitk.GetArrayFromImage(fixed_mask).astype(bool)

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


# feature_extractor = OrientedHistogramFeatureExtrator(device=DEVICE)
# parameter_guesser = ParameterGuesser(
#     database_filepath="/datalake/learn2reg/param_sampling_v2.sqlite",
#     parameters_to_guess=(
#         # "tau",
#         "tau_level_decay",
#         "tau_iteration_decay",
#         # "sigma_x",
#         # "sigma_y",
#         # "sigma_z",
#         # "sigma_level_decay",
#         # "sigma_iteration_decay",
#         # "n_levels",
#     ),
# )
# parameter_guesser.fit()

from vroc.decay import half_life_to_lambda

params = {
    "iterations": 100,
    "tau": 1.00,  # 2.25,
    # "tau_level_decay": 0.0,
    # "tau_iteration_decay": 0.0,
    "sigma_x": 1.25,
    "sigma_y": 1.25,
    "sigma_z": 1.25,
    # "sigma_level_decay": half_life_to_lambda(-32),
    # "sigma_iteration_decay": 0.0, #half_life_to_lambda(3200),
    "n_levels": 3,
}

registration = VrocRegistration(
    roi_segmenter=None,
    # feature_extractor=feature_extractor,
    # parameter_guesser=parameter_guesser,
    device=DEVICE,
)

tres_before = []
tres_after = []
smoothnesses = []
t_start = time.time()


for case in range(1, 2):
    fixed_landmarks = read_landmarks(
        f"{ROOT_DIR}/{FOLDER}/keypointsTr/LungCT_{case:04d}_0000.csv",
        sep=",",
    )
    moving_landmarks = read_landmarks(
        f"{ROOT_DIR}/{FOLDER}/keypointsTr/LungCT_{case:04d}_0001.csv",
        sep=",",
    )

    (
        moving_image,
        fixed_image,
        moving_mask,
        fixed_mask,
        image_spacing,
        reference_image,
    ) = load(
        moving_image_filepath=f"{ROOT_DIR}/{FOLDER}/imagesTr/LungCT_{case:04d}_0001.nii.gz",
        fixed_image_filepath=f"{ROOT_DIR}/{FOLDER}/imagesTr/LungCT_{case:04d}_0000.nii.gz",
        moving_mask_filepath=f"{ROOT_DIR}/{FOLDER}/masksTr/LungCT_{case:04d}_0001.nii.gz",
        fixed_mask_filepath=f"{ROOT_DIR}/{FOLDER}/masksTr/LungCT_{case:04d}_0000.nii.gz",
    )

    # moving_mask = binary_dilation(moving_mask.astype(np.uint8), iterations=1).astype(
    #     bool
    # )
    # fixed_mask = binary_dilation(fixed_mask.astype(np.uint8), iterations=1).astype(bool)
    # moving_mask = moving_mask.astype(bool)
    # fixed_mask = fixed_mask.astype(bool)

    # union_mask = moving_mask | fixed_mask

    moving_keypoints = torch.as_tensor(moving_landmarks[np.newaxis], device=DEVICE)
    fixed_keypoints = torch.as_tensor(fixed_landmarks[np.newaxis], device=DEVICE)

    registration_result = registration.register(
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_mask=moving_mask,
        fixed_mask=fixed_mask,
        register_affine=True,
        affine_loss_function=mse_loss,
        affine_step_size=0.01,
        affine_iterations=300,
        force_type="demons",
        gradient_type="dual",
        valid_value_range=(-1024, 3071),
        early_stopping_delta=0.00001,
        early_stopping_window=None,
        default_parameters=params,
        debug=True,
        debug_step_size=1,
        return_as_tensor=False,
    )

    # model = DemonsVectorFieldBooster(shape=(224, 192, 224), n_iterations=4).to(DEVICE)
    # optimizer = Adam(model.parameters(), lr=5e-4)
    #
    # registration_result = registration.register_and_train_boosting(
    #     # boosting specific kwargs
    #     model=model,
    #     optimizer=optimizer,
    #     n_iterations=400,
    #     moving_keypoints=moving_keypoints,
    #     fixed_keypoints=fixed_keypoints,
    #     # registration specific kwargs
    #     moving_image=moving_image,
    #     fixed_image=fixed_image,
    #     moving_mask=moving_mask,
    #     fixed_mask=fixed_mask,
    #     image_spacing=image_spacing,
    #     register_affine=True,
    #     affine_loss_fn=ncc_loss,
    #     force_type="demons",
    #     gradient_type="active",
    #     valid_value_range=(-1024, 3071),
    #     early_stopping_delta=0.00001,
    #     early_stopping_window=None,
    #     default_parameters=params,
    #     debug=False,
    # )

    vector_field = registration_result.composed_vector_field

    smoothness = calculate_l2r_smoothness(vector_field, mask=fixed_mask)
    smoothnesses.append(smoothness)

    # warped_moving_image = registration_result.warped_moving_image.swapaxes(0, 2)
    # warped_moving_image = sitk.GetImageFromArray(warped_moving_image)
    # warped_moving_image.CopyInformation(reference_image)
    # sitk.WriteImage(
    #     warped_moving_image, str(OUTPUT_FOLDER / f"warped_image_{case}_{case}.nii")
    # )
    #
    # write_nlst_vector_field(
    #     vector_field,
    #     case=f"{case:04d}",
    #     output_folder=OUTPUT_FOLDER,
    # )

    tre_before = compute_tre_numpy(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=None,
        image_spacing=image_spacing,
    )
    tre_affine = compute_tre_numpy(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=registration_result.vector_fields[0],
        image_spacing=image_spacing,
    )
    tre_vroc = compute_tre_numpy(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=registration_result.composed_vector_field,
        image_spacing=image_spacing,
    )

    if any(tre > tre_before.mean() for tre in (tre_affine.mean(), tre_vroc.mean())):
        level = logging.ERROR
    else:
        level = logging.INFO

    logger.log(
        level,
        f"NLST_0{case}: "
        f"TRE: "
        f"before={tre_before.mean():.3f} / "
        f"affine={tre_affine.mean():.3f} / "
        f"vroc={tre_vroc.mean():.3f}, "
        f"smoothness={smoothness:.3f}",
    )

    tres_before.append(tre_before.mean())
    tres_after.append(tre_vroc.mean())


print(f"before: mean TRE={np.mean(tres_before)}, std TRE={np.std(tres_before)}")
print(f"after: mean TRE={np.mean(tres_after)}, std TRE={np.std(tres_after)}")
print(f"after: mean smoothness={np.mean(smoothnesses)}, std TRE={np.std(smoothnesses)}")

print(f"run took {time.time() - t_start}")
