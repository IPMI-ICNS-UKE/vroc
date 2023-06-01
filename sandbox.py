import logging
import time

import numpy as np

from vroc.logger import init_fancy_logging
from vroc.loss import ncc_loss
from vroc.registration import VrocRegistration

if __name__ == "__main__":
    init_fancy_logging()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.getLogger("vroc").setLevel(logging.DEBUG)

    params = {
        "iterations": 800,
        "tau": 2.00,
        "tau_level_decay": 0.0,
        "tau_iteration_decay": 0.0,
        "sigma_x": 1.25,
        "sigma_y": 1.25,
        "sigma_z": 1.25,
        "sigma_level_decay": 0.0,
        "sigma_iteration_decay": 0.0,
        "n_levels": 3,
        "largest_scale_factor": 1,
    }

    registration = VrocRegistration(
        device="cuda:0",
    )

    t_start = time.time()

    moving_image = np.random.random((128, 128, 128)).astype(np.float32)
    fixed_image = np.random.random((128, 128, 128)).astype(np.float32)

    moving_mask = np.ones_like(moving_image, dtype=bool)
    fixed_mask = np.ones_like(fixed_image, dtype=bool)

    reg_result = registration.register(
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_mask=moving_mask,
        fixed_mask=fixed_mask,
        image_spacing=(1.0, 1.0, 1.0),
        register_affine=False,
        affine_loss_function=ncc_loss,
        force_type="demons",
        gradient_type="dual",
        valid_value_range=(-1024, 3071),
        early_stopping_delta=0.00,
        early_stopping_window=100,
        default_parameters=params,
        mode="standard",
        affine_enable_rotation=True,
        affine_enable_scaling=True,
        affine_enable_shearing=True,
        affine_enable_translation=True,
    )

    logger.info(f"run took {time.time() - t_start}")
