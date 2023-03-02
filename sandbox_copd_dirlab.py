import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from matplotlib.animation import FFMpegWriter
from scipy.ndimage.morphology import binary_dilation
from torch.optim import Adam

from vroc import models
from vroc.decay import half_life_to_lambda
from vroc.helper import compute_tre_numpy, read_landmarks
from vroc.keypoints import extract_keypoints
from vroc.logger import FancyFormatter
from vroc.loss import mse_loss, ncc_loss
from vroc.registration import VrocRegistration

# for reproducibility
random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True, warn_only=True)
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(FancyFormatter())
logging.basicConfig(handlers=[handler])

logging.getLogger(__name__).setLevel(logging.DEBUG)
logging.getLogger("vroc").setLevel(logging.DEBUG)
logging.getLogger("vroc.affine").setLevel(logging.INFO)
logging.getLogger("vroc.models.VarReg").setLevel(logging.INFO)

ROOT_DIR = (
    Path("/home/tsentker/data"),
    Path("/datalake"),
    Path("/datalake/learn2reg"),
)
FOLDER = "copd_dirlab2022"

ROOT_DIR = next(p for p in ROOT_DIR if (p / FOLDER).exists())

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

    image_spacing = fixed_image.GetSpacing()

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


# feature_extractor = OrientedHistogramFeatureExtrator(device="cuda:0")
# parameter_guesser = ParameterGuesser(
#     database_filepath="/datalake/learn2reg/best_parameters.sqlite",
#     parameters_to_guess=('sigma_x', 'sigma_y', 'sigma_z')
# )
# parameter_guesser.fit()


registration = VrocRegistration(
    roi_segmenter=None,
    feature_extractor=None,
    parameter_guesser=None,
    device="cuda:0",
)

tres_before = []
tres_after = []
t_start = time.time()
for case in range(8, 9):
    moving_landmarks = read_landmarks(
        f"{ROOT_DIR}/{FOLDER}/data/copd{case:02d}/extremePhases/landmarks_e.txt",
        sep="\t",
    )
    fixed_landmarks = read_landmarks(
        f"{ROOT_DIR}/{FOLDER}/data/copd{case:02d}/extremePhases/landmarks_i.txt",
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
        moving_image_filepath=f"{ROOT_DIR}/{FOLDER}/data/copd{case:02d}/Images/phase_e.mha",
        fixed_image_filepath=f"{ROOT_DIR}/{FOLDER}/data/copd{case:02d}/Images/phase_i.mha",
        moving_mask_filepath=f"{ROOT_DIR}/{FOLDER}/data/copd{case:02d}/segmentation/mask_e.mha",
        fixed_mask_filepath=f"{ROOT_DIR}/{FOLDER}/data/copd{case:02d}/segmentation/mask_i.mha",
    )

    moving_mask = binary_dilation(moving_mask.astype(np.uint8), iterations=1).astype(
        bool
    )
    fixed_mask = binary_dilation(fixed_mask.astype(np.uint8), iterations=1).astype(bool)

    # moving_mask = moving_mask.astype(bool)
    # fixed_mask = fixed_mask.astype(bool)
    # union_mask = moving_mask | fixed_mask

    # debug = False
    # reg_result = registration.register(
    #     moving_image=moving_image,
    #     fixed_image=fixed_image,
    #     moving_mask=moving_mask,
    #     fixed_mask=fixed_mask,
    #     image_spacing=image_spacing,
    #     register_affine=False,
    #     affine_loss_function=ncc_loss,
    #     force_type="demons",
    #     gradient_type="active",
    #     valid_value_range=(-1024, 3071),
    #     early_stopping_delta=0.00,
    #     early_stopping_window=100,
    #     default_parameters=params,
    #     # debug=True,
    #     # debug_output_folder="/home/tsentker/data/copd_dirlab2022/data/copd08/debug",
    #     # debug_step_size=10,
    # )

    # moving_keypoints, fixed_keypoints = extract_keypoints(
    #     moving_image=torch.as_tensor(moving_image[None, None], device='cuda:0'),
    #     fixed_image=torch.as_tensor(fixed_image[None, None], device='cuda:0'),
    #     fixed_mask=torch.as_tensor(fixed_mask[None, None], device='cuda:0'),
    # )
    # keypoints_tre = torch.sum((moving_keypoints - fixed_keypoints) ** 2, dim=-1) ** 0.5
    # print(f'{keypoints_tre.mean()=}')
    moving_keypoints = torch.as_tensor(moving_landmarks[None], device="cuda:0")
    fixed_keypoints = torch.as_tensor(fixed_landmarks[None], device="cuda:0")

    params_init = {
        "iterations": 0,
        "tau": 2.25,
        "tau_iteration_decay": 0.0,  # half_life_to_lambda(800),
        "sigma_x": 2.00,
        "sigma_y": 2.00,
        "sigma_z": 2.00,
        "n_levels": 3,
        "largest_scale_factor": 1.0,
    }

    params_finetune = {
        "iterations": 200,
        "tau": 1.00,
        "tau_iteration_decay": 0.0,  # half_life_to_lambda(800),
        "sigma_x": 1.00,
        "sigma_y": 1.00,
        "sigma_z": 1.00,
        "n_levels": 2,
        "largest_scale_factor": 1.0,
    }

    debug = True
    if debug:
        debug_folder = Path(f"/datalake/copd_dirlab2022/debug/{int(time.time())}")
    else:
        debug_folder = None

    tre_before = compute_tre_numpy(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=None,
        image_spacing=image_spacing,
    )

    model = models.DemonsVectorFieldBooster5(n_iterations=1).to("cuda:0")
    optimizer = Adam(model.parameters(), lr=1e-3)

    registration_result = registration.register_and_train_boosting(
        # boosting specific kwargs
        model=model,
        optimizer=optimizer,
        n_iterations=400,
        moving_keypoints=moving_keypoints,
        fixed_keypoints=fixed_keypoints,
        image_loss_function="mse",
        boost_scale=1 / 8,
        # loss weights
        keypoint_loss_weight=1.0,
        label_loss_weight=0.0,
        smoothness_loss_weight=0.0,
        image_loss_weight=0.0,
        # registration specific kwargs
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_mask=moving_mask,
        fixed_mask=fixed_mask,
        use_masks=True,
        image_spacing=image_spacing,
        register_affine=False,
        affine_loss_function=mse_loss,
        affine_step_size=0.01,
        affine_iterations=300,
        force_type="demons",
        gradient_type="active",
        valid_value_range=(-1000, 3071),
        early_stopping_delta=0.00001,
        early_stopping_window=None,
        default_parameters=params_init,
        debug=debug,
        debug_output_folder=debug_folder / "init",
        debug_step_size=50,
    )

    tre_after_init = compute_tre_numpy(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=registration_result.composed_vector_field,
        image_spacing=image_spacing,
        snap_to_voxel=True,
    )

    model = models.DemonsVectorFieldBooster5(n_iterations=1).to("cuda:0")
    optimizer = Adam(model.parameters(), lr=1e-3)

    registration_result = registration.register_and_train_boosting(
        # boosting specific kwargs
        model=model,
        optimizer=optimizer,
        n_iterations=200,
        moving_keypoints=moving_keypoints,
        fixed_keypoints=fixed_keypoints,
        image_loss_function="mse",
        boost_scale=1 / 2,
        # loss weights
        keypoint_loss_weight=1.0,
        label_loss_weight=0.0,
        smoothness_loss_weight=0.0,
        image_loss_weight=0.0,
        # registration specific kwargs
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_mask=moving_mask,
        fixed_mask=fixed_mask,
        use_masks=True,
        image_spacing=image_spacing,
        initial_vector_field=registration_result.composed_vector_field,
        register_affine=False,
        affine_loss_function=mse_loss,
        affine_step_size=0.01,
        affine_iterations=300,
        force_type="demons",
        gradient_type="active",
        valid_value_range=(-1000, 3071),
        early_stopping_delta=0.00001,
        early_stopping_window=None,
        default_parameters=params_init,
        debug=debug,
        debug_output_folder=debug_folder / "fine",
        debug_step_size=50,
    )

    # registration_result = registration.register(
    #     moving_image=registration_result.moving_image,
    #     fixed_image=registration_result.fixed_image,
    #     moving_mask=registration_result.warped_moving_mask,
    #     fixed_mask=registration_result.fixed_mask,
    #     use_masks=True,
    #     image_spacing=image_spacing,
    #     initial_vector_field=registration_result.composed_vector_field,
    #     register_affine=False,
    #     affine_loss_function=mse_loss,
    #     affine_step_size=0.01,
    #     affine_iterations=300,
    #     force_type="demons",
    #     gradient_type="active",
    #     valid_value_range=(-1000, 3071),
    #     early_stopping_delta=0.00001,
    #     early_stopping_window=None,
    #     default_parameters=params_finetune,
    # )
    #
    #
    tre_after_finetune = compute_tre_numpy(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=registration_result.composed_vector_field,
        image_spacing=image_spacing,
        snap_to_voxel=True,
    )

    tres = {
        "total_before": tre_before,
        "total_after_init": tre_after_init,
        "total_after_finetune": tre_after_finetune,
    }

    print(f"copd_dirlab2022_0{case}:")
    for tre_name, tre in tres.items():
        print(f"{tre_name:<25}: {tre.mean():5.2f} ± {tre.std():5.2f}")

    tres_before.append(tres["total_before"].mean())
    tres_after.append(tres["total_after_finetune"].mean())

print(f"before: mean TRE={np.mean(tres_before)}, std TRE={np.std(tres_before)}")
print(f"after: mean TRE={np.mean(tres_after)}, std TRE={np.std(tres_after)}")

print(f"run took {time.time() - t_start}")
