import logging
import pickle
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage.morphology import binary_dilation

from vroc.helper import read_landmarks
from vroc.logger import init_fancy_logging
from vroc.loss import ncc_loss
from vroc.registration import VrocRegistration

init_fancy_logging()

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
logging.getLogger("vroc").setLevel(logging.DEBUG)
logging.getLogger("vroc.affine").setLevel(logging.INFO)
logging.getLogger("vroc.models.VarReg").setLevel(logging.INFO)

ROOT_DIR = (
    Path("/home/tsentker/data/learn2reg"),
    Path("/datalake/learn2reg"),
)
ROOT_DIR = next(p for p in ROOT_DIR if p.exists())
FOLDER = "NLST_Validation"

OUTPUT_FOLDER = Path(f"{ROOT_DIR}/{FOLDER}/predictions/non_boosted")
OUTPUT_FOLDER.mkdir(exist_ok=True)

DEVICE = "cuda:1"


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
    device=DEVICE,
)


for case in range(101, 111):
    fixed_landmarks = read_landmarks(
        f"{ROOT_DIR}/{FOLDER}/keypointsTr/NLST_{case:04d}_0000.csv",
        sep=" ",
    )
    moving_landmarks = read_landmarks(
        f"{ROOT_DIR}/{FOLDER}/keypointsTr/NLST_{case:04d}_0001.csv",
        sep=" ",
    )

    (
        moving_image,
        fixed_image,
        moving_mask,
        fixed_mask,
        image_spacing,
        reference_image,
    ) = load(
        moving_image_filepath=f"{ROOT_DIR}/{FOLDER}/imagesTr/NLST_{case:04d}_0001.nii.gz",
        fixed_image_filepath=f"{ROOT_DIR}/{FOLDER}/imagesTr/NLST_{case:04d}_0000.nii.gz",
        moving_mask_filepath=f"{ROOT_DIR}/{FOLDER}/masksTr/NLST_{case:04d}_0001.nii.gz",
        fixed_mask_filepath=f"{ROOT_DIR}/{FOLDER}/masksTr/NLST_{case:04d}_0000.nii.gz",
    )

    moving_mask = binary_dilation(moving_mask.astype(np.uint8), iterations=1).astype(
        bool
    )
    fixed_mask = binary_dilation(fixed_mask.astype(np.uint8), iterations=1).astype(bool)

    moving_keypoints = torch.as_tensor(moving_landmarks[np.newaxis], device=DEVICE)
    fixed_keypoints = torch.as_tensor(fixed_landmarks[np.newaxis], device=DEVICE)

    registration_result = registration.register(
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_mask=moving_mask,
        fixed_mask=fixed_mask,
        image_spacing=image_spacing,
        register_affine=True,
        affine_loss_function=ncc_loss,
        affine_step_size=0.1,
        affine_iterations=300,
        force_type="demons",
        gradient_type="active",
        valid_value_range=(-1024, 3071),
        early_stopping_delta=0.00001,
        early_stopping_window=None,
        default_parameters=params,
        debug=False,
        debug_step_size=1,
        return_as_tensor=True,
    )

    registration_result.moving_keypoints = moving_keypoints
    registration_result.fixed_keypoints = fixed_keypoints

    registration_result.to("cpu")
    registration_result.vector_fields = None
    registration_result.warped_moving_image = None
    registration_result.warped_moving_mask = None
    registration_result.warped_affine_moving_image = None
    registration_result.warped_affine_moving_mask = None

    with open(f"{ROOT_DIR}/{FOLDER}/boosting_samples/NLST_{case:04d}.pkl", "wb") as f:
        pickle.dump(registration_result, f)
