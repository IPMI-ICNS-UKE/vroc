import logging
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from matplotlib.animation import FFMpegWriter
from scipy.ndimage.morphology import binary_dilation

from vroc.dataset import Lung4DCTRegistrationDataset
from vroc.helper import compute_tre_numpy, read_landmarks
from vroc.logger import init_fancy_logging
from vroc.loss import ncc_loss
from vroc.registration import VrocRegistration

if __name__ == "__main__":

    init_fancy_logging()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.getLogger("vroc").setLevel(logging.INFO)
    logging.getLogger("vroc.models").setLevel(logging.INFO)

    ROOT_DIR = (
        Path("/home/tsentker/data"),
        Path("/datalake"),
    )
    ROOT_DIR = next(p for p in ROOT_DIR if p.exists())
    DATASET = "dirlab_4dct"
    DATASET = "dirlab_copdgene"

    DATA_FOLDER = ROOT_DIR / DATASET / "converted"

    device = "cuda:0"

    dataset = Lung4DCTRegistrationDataset.from_folder(
        DATA_FOLDER, phases=(0, 5), output_image_spacing=None
    )

    # feature_extractor = OrientedHistogramFeatureExtrator(device="cuda:0")
    # parameter_guesser = ParameterGuesser(
    #     database_filepath="/datalake/learn2reg/best_parameters.sqlite",
    #     parameters_to_guess=('sigma_x', 'sigma_y', 'sigma_z')
    # )
    # parameter_guesser.fit()

    params = {
        "iterations": 100,
        "tau": 2.25,
        "tau_level_decay": 0.0,
        "tau_iteration_decay": 0.0,
        "sigma_x": 1.25,
        "sigma_y": 1.25,
        "sigma_z": 1.25,
        "sigma_level_decay": 0.0,
        "sigma_iteration_decay": 0.0,
        "n_levels": 3,
        "largest_scale_factor": 1.0,
    }
    registration = VrocRegistration(
        roi_segmenter=None,
        feature_extractor=None,
        parameter_guesser=None,
        device="cuda:0",
    )

    tres_before = []
    tres_after = []
    t_start = time.time()
    for data in dataset:
        moving_phase = 5  # == max exhale
        fixed_phase = 0  # == max inhale

        moving_image = data["images"][moving_phase]["data"]
        fixed_image = data["images"][fixed_phase]["data"]

        moving_mask = data["masks"][moving_phase]["data"]
        fixed_mask = data["masks"][fixed_phase]["data"]

        moving_landmarks, fixed_landmarks = data["landmarks"][
            (moving_phase, fixed_phase)
        ]
        image_spacing = data["meta"]["image_spacing"]

        moving_mask = binary_dilation(
            moving_mask.astype(np.uint8), iterations=1
        ).astype(bool)
        fixed_mask = binary_dilation(fixed_mask.astype(np.uint8), iterations=1).astype(
            bool
        )

        reg_result = registration.register(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            image_spacing=image_spacing,
            register_affine=False,
            affine_loss_function=ncc_loss,
            force_type="demons",
            gradient_type="dual",
            valid_value_range=(-1024, 3071),
            early_stopping_delta=0.00,
            early_stopping_window=100,
            default_parameters=params,
            debug=False,
            debug_output_folder=DATA_FOLDER.parent / "debug",
            debug_step_size=1,
        )

        tre_before = compute_tre_numpy(
            moving_landmarks=moving_landmarks,
            fixed_landmarks=fixed_landmarks,
            vector_field=None,
            image_spacing=image_spacing,
        )
        tre_after = compute_tre_numpy(
            moving_landmarks=moving_landmarks,
            fixed_landmarks=fixed_landmarks,
            vector_field=reg_result.composed_vector_field,
            image_spacing=image_spacing,
            snap_to_voxel=True,
        )

        tre_before = np.mean(tre_before)
        tre_after = np.mean(tre_after)
        tre_observer = data["meta"]["observer_tre_mean"]
        logger.info(f"Patient {data['folder']}:")
        logger.info(f"tre_before   = {tre_before:>5.2f}")
        logger.info(f"tre_after    = {tre_after:>5.2f}")
        logger.info(f"tre_observer = {tre_observer:>5.2f}")

        tres_before.append(np.mean(tre_before))
        tres_after.append(np.mean(tre_after))

    logger.info(
        f"mean_tre_before = {np.mean(tres_before):5.2f} ± {np.std(tres_before):5.2f}"
    )
    logger.info(
        f"mean_tre_after  = {np.mean(tres_after):5.2f} ± {np.std(tres_after):5.2f}"
    )
    logger.info(f"run took {time.time() - t_start}")
