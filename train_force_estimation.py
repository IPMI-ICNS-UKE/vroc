import logging
import os
import random
import time
from pathlib import Path

import aim
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from ipmi.deeplearning.trainer import BestModelSaver, MetricType
from matplotlib.animation import FFMpegWriter
from scipy.ndimage.morphology import binary_dilation
from torch.optim import Adam

from vroc import models
from vroc.blocks import GaussianSmoothing3d, SpatialTransformer
from vroc.decay import half_life_to_lambda
from vroc.helper import compute_tre_numpy, read_landmarks, write_registration_result
from vroc.interpolation import match_vector_field, resize
from vroc.keypoints import extract_keypoints
from vroc.logger import FancyFormatter
from vroc.loss import (
    TRELoss,
    WarpedMSELoss,
    mse_loss,
    ncc_loss,
    smooth_vector_field_loss,
)
from vroc.models import VariationalRegistrationBooster
from vroc.registration import RegistrationResult, VrocRegistration

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("vroc").setLevel(logging.DEBUG)
logging.getLogger("vroc.affine").setLevel(logging.INFO)
logging.getLogger("vroc.models").setLevel(logging.INFO)

ROOT_DIR = (
    Path("/home/tsentker/data"),
    Path("/datalake"),
    Path("/datalake/learn2reg"),
)
FOLDER = "copd_dirlab2022"

ROOT_DIR = next(p for p in ROOT_DIR if (p / FOLDER).exists())

device = "cuda:0"

OUTPUT_FOLDER = Path(f"{ROOT_DIR}/{FOLDER}/output")


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

GRADIENT_TYPE = "dual"
registration = VrocRegistration(
    roi_segmenter=None,
    feature_extractor=None,
    parameter_guesser=None,
    device="cuda:0",
)

tres_before = []
tres_after = []
t_start = time.time()


train_cases = [1, 2, 3, 4, 5]
test_cases = [6, 7, 8, 9, 10]
n_epochs = 100000

update_interval = 10

if n_epochs > 0:
    repo = aim.Repo(str(OUTPUT_FOLDER / "aim"), init=True)
    run = aim.Run(repo=repo)

    models = {
        f"level_{i_level}": models.DemonsForceEstimator(gradient_type=GRADIENT_TYPE).to(
            "cuda:0"
        )
        for i_level in range(3)
    }

    optimizers = {
        level: Adam(model.parameters(), lr=1e-4) for (level, model) in models.items()
    }

    model_savers = {
        level: BestModelSaver(
            tracked_metrics={"epoch": MetricType.LARGER_IS_BETTER},
            model=model,
            model_name=level,
            optimizer=optimizers[level],
            output_folder=OUTPUT_FOLDER / "models" / run.hash,
            top_k=10,
        )
        for (level, model) in models.items()
    }

for i_epoch in range(n_epochs):
    for case in train_cases:
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

        moving_mask = binary_dilation(
            moving_mask.astype(np.uint8), iterations=1
        ).astype(bool)
        fixed_mask = binary_dilation(fixed_mask.astype(np.uint8), iterations=1).astype(
            bool
        )

        # moving_mask = moving_mask.astype(bool)
        # fixed_mask = fixed_mask.astype(bool)
        # union_mask = moving_mask | fixed_mask

        moving_keypoints = torch.as_tensor(moving_landmarks[None], device="cuda:0")
        fixed_keypoints = torch.as_tensor(fixed_landmarks[None], device="cuda:0")

        params = {
            "iterations": 200,
            "tau": 2.25,
            "tau_iteration_decay": 0.0,  # half_life_to_lambda(800),
            "sigma_x": 2.00,
            "sigma_y": 2.00,
            "sigma_z": 2.00,
            "n_levels": 3,
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

        registration_result = registration.register(
            # registration specific kwargs
            moving_image=moving_image,
            fixed_image=fixed_image,
            # moving_mask=moving_mask,
            # fixed_mask=fixed_mask,
            use_masks=True,
            image_spacing=image_spacing,
            register_affine=False,
            affine_loss_function=mse_loss,
            affine_step_size=0.01,
            affine_iterations=300,
            force_type="demons",
            gradient_type=GRADIENT_TYPE,
            valid_value_range=(-1000, 3071),
            early_stopping_delta=0.0001,
            early_stopping_window=None,
            default_parameters=params,
            yield_each_step=True,
        )

        tre_loss = TRELoss(apply_sqrt=True)

        full_image_shape = fixed_image.shape

        regularization = GaussianSmoothing3d(
            sigma=(params["sigma_x"], params["sigma_y"], params["sigma_z"]),
            sigma_cutoff=(2.0, 2.0, 2.0),
            force_same_size=True,
        ).to(device)

        for i_step, step in enumerate(registration_result):
            if i_step % update_interval > 0 or isinstance(step, RegistrationResult):
                continue

            level = step["level"]
            optimizer = optimizers[f"level_{level}"]
            model = models[f"level_{level}"]
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=True):
                estimated_forces, demon_forces = model(
                    moving_image=step["moving_image"],
                    fixed_image=step["fixed_image"],
                    moving_mask=step["moving_mask"],
                    fixed_mask=step["fixed_mask"],
                    vector_field=step["vector_field_before"],
                    image_spacing=None,
                )

                vector_field_after_boost = (
                    step["vector_field_before"] + params["tau"] * estimated_forces
                )
                vector_field_after_boost = regularization(vector_field_after_boost)

                mse_forces = F.mse_loss(estimated_forces, demon_forces)

                mse_loss_func = WarpedMSELoss()

                mse_after = mse_loss_func(
                    fixed_image=step["fixed_image"],
                    moving_image=step["moving_image"],
                    vector_field=step["vector_field_after"],
                    fixed_mask=step["fixed_mask"],
                )
                mse_after_boost = mse_loss_func(
                    fixed_image=step["fixed_image"],
                    moving_image=step["moving_image"],
                    vector_field=vector_field_after_boost,
                    fixed_mask=step["fixed_mask"],
                )

                mse_difference = mse_after_boost - mse_after

                vector_field_before = match_vector_field(
                    vector_field=step["vector_field_before"],
                    image_shape=full_image_shape,
                )
                vector_field_after = match_vector_field(
                    vector_field=step["vector_field_after"],
                    image_shape=full_image_shape,
                )
                vector_field_after_boost = match_vector_field(
                    vector_field=vector_field_after_boost, image_shape=full_image_shape
                )

                tre_before = tre_loss(
                    vector_field=vector_field_before,
                    moving_landmarks=moving_keypoints,
                    fixed_landmarks=fixed_keypoints,
                    image_spacing=image_spacing,
                )
                tre_after = tre_loss(
                    vector_field=vector_field_after,
                    moving_landmarks=moving_keypoints,
                    fixed_landmarks=fixed_keypoints,
                    image_spacing=image_spacing,
                )
                tre_after_boost = tre_loss(
                    vector_field=vector_field_after_boost,
                    moving_landmarks=moving_keypoints,
                    fixed_landmarks=fixed_keypoints,
                    image_spacing=image_spacing,
                )

                fixed_mask = resize(step["fixed_mask"], output_shape=full_image_shape)

                tre_difference = tre_after - tre_before
                tre_difference_boost = tre_after_boost - tre_before
                tre_improvement = tre_after_boost - tre_after
                mse_improvement = mse_after_boost - mse_after

            logger.info(
                f"[{i_epoch}/{i_step}] Case {case}, {tre_before=:.2f}, {tre_after=:.2f}, {tre_after_boost=:.2f}, {tre_after_boost-tre_after=:.2f}, {mse_improvement=:.6f}"
            )

            if i_epoch < 0:
                loss = mse_forces
            else:
                loss = mse_improvement

            logger.info(f"[{i_epoch}/{i_step}] Case {case}, {loss.item()=:.6f}")
            loss.backward()
            optimizer.step()

            track_step = i_step * (i_epoch + 1)

            run.track(
                mse_forces,
                name="mse_forces",
                step=track_step,
                epoch=i_epoch,
                context={"case": case},
            )
            run.track(
                tre_before,
                name="tre_before",
                step=track_step,
                epoch=i_epoch,
                context={"case": case},
            )
            run.track(
                tre_after,
                name="tre_after",
                step=track_step,
                epoch=i_epoch,
                context={"case": case},
            )
            run.track(
                tre_after_boost,
                name="tre_after_boost",
                step=track_step,
                epoch=i_epoch,
                context={"case": case},
            )
            run.track(
                tre_improvement,
                name="tre_improvement",
                step=track_step,
                epoch=i_epoch,
                context={"case": case},
            )
            run.track(
                mse_improvement,
                name="mse_improvement",
                step=track_step,
                epoch=i_epoch,
                context={"case": case},
            )

    for model_saver in model_savers.values():
        model_saver.track({"epoch": i_epoch}, step=i_epoch)


for case in test_cases:
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

    moving_keypoints = torch.as_tensor(moving_landmarks[None], device="cuda:0")
    fixed_keypoints = torch.as_tensor(fixed_landmarks[None], device="cuda:0")

    params = {
        "iterations": 800,
        "tau": 2.25,
        "tau_iteration_decay": 0.0,  # half_life_to_lambda(800),
        "sigma_x": 2.00,
        "sigma_y": 2.00,
        "sigma_z": 2.00,
        "n_levels": 3,
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
    ).mean()

    registration_result = registration.register(
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
        default_parameters=params,
        mode="force_estimation",
    )

    tre_after = compute_tre_numpy(
        vector_field=registration_result.composed_vector_field,
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        image_spacing=image_spacing,
    ).mean()

    write_registration_result(
        registration_result=registration_result,
        output_folder=f"/datalake/copd_dirlab2022/output/case_{case:02d}",
    )

    logger.info(f"Case {case}, {tre_before=:.2f}, {tre_after=:.2f}")
