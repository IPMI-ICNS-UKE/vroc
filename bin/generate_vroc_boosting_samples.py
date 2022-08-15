from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import nevergrad as ng
import numpy as np
import torch
import typer
from scipy.ndimage import binary_dilation
from torch.optim import Adam

import vroc.database.models as orm
from vroc.dataset import NLSTDataset
from vroc.helper import compute_tre_numpy
from vroc.hyperopt_database.client import DatabaseClient
from vroc.logger import LogFormatter
from vroc.loss import TRELoss
from vroc.metrics import root_mean_squared_error
from vroc.models import VectorFieldBooster
from vroc.registration import VrocRegistration

logger = logging.getLogger(__name__)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(LogFormatter())
logging.basicConfig(handlers=[handler])

logging.getLogger(__name__).setLevel(logging.INFO)
logging.getLogger("vroc").setLevel(logging.INFO)


OUTPUT_PATH = Path("/datalake/learn2reg/NLST/detailed_boosting_data")
OUTPUT_PATH.mkdir(exist_ok=True)

DEVICE = "cuda:1"
ITERATIONS_PER_IMAGE = 20

PARAM_CONFIG = ng.p.Instrumentation(
    # iterations=ng.p.Scalar(lower=100, upper=1000).set_integer_casting(),
    iterations=ng.p.Constant(1000),
    tau=ng.p.Scalar(lower=0.5, upper=10.0),
    sigma_x=ng.p.Scalar(lower=0.5, upper=5.0),
    sigma_y=ng.p.Scalar(lower=0.5, upper=5.0),
    sigma_z=ng.p.Scalar(lower=0.5, upper=5.0),
    n_levels=ng.p.Scalar(lower=1, upper=4).set_integer_casting(),
)

train_dataset = NLSTDataset(
    "/datalake/learn2reg/NLST", i_worker=None, n_worker=None, dilate_masks=1
)

ng_optimizer = ng.optimizers.RandomSearch(parametrization=PARAM_CONFIG)

for data in train_dataset:
    # get images/masks/... and remove color channel
    moving_image = data["moving_image"][0]
    fixed_image = data["fixed_image"][0]
    moving_mask = data["moving_mask"][0]
    fixed_mask = data["fixed_mask"][0]
    moving_keypoints = data["moving_keypoints"]
    fixed_keypoints = data["fixed_keypoints"]
    image_spacing = data["image_spacing"]

    moving_mask = binary_dilation(moving_mask.astype(np.uint8), iterations=1).astype(
        np.uint8
    )
    fixed_mask = binary_dilation(fixed_mask.astype(np.uint8), iterations=1).astype(
        np.uint8
    )

    union_mask = moving_mask | fixed_mask
    moving_mask = union_mask
    fixed_mask = union_mask

    # calculate TRE before registration
    tre_before = compute_tre_numpy(
        moving_landmarks=moving_keypoints,
        fixed_landmarks=fixed_keypoints,
        vector_field=None,
        image_spacing=image_spacing,
    )
    tre_before_mean = tre_before.mean()
    tre_before_std = tre_before.std()

    for i_sample in range(ITERATIONS_PER_IMAGE):
        filename = (
            f"{data['moving_image_name'][-21:-12]}_vector_field_{i_sample:04d}.pkl"
        )

        if (OUTPUT_PATH / filename).exists():
            continue

        # sample random registration params
        params = ng_optimizer.ask()

        registration = VrocRegistration(
            roi_segmenter=None,
            feature_extractor=None,
            parameter_guesser=None,
            device=DEVICE,
        )
        result = registration.register(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            register_affine=True,
            valid_value_range=(-1024, 3071),
            early_stopping_delta=0.001,
            early_stopping_window=100,
            default_parameters=params.kwargs,
        )
        # calculate TRE after registration, but w/o boosting
        tre_after_affine = compute_tre_numpy(
            moving_landmarks=moving_keypoints,
            fixed_landmarks=fixed_keypoints,
            vector_field=result.vector_fields[0],
            image_spacing=image_spacing,
        )
        tre_after_varreg = compute_tre_numpy(
            moving_landmarks=moving_keypoints,
            fixed_landmarks=fixed_keypoints,
            vector_field=result.composed_vector_field,
            image_spacing=image_spacing,
        )

        # write data to disk
        with open(OUTPUT_PATH / filename, "wb") as f:
            pickle.dump(
                {
                    "affine_vector_field": result.vector_fields[0],
                    "varreg_vector_field": result.vector_fields[1],
                    "parameters": params.kwargs,
                    "tre_before": tre_before,
                    "tre_after_affine": tre_after_affine,
                    "tre_after_varreg": tre_after_varreg,
                },
                f,
            )

            logger.info(
                f"{filename}: "
                f"TRE: before/after affine/after varreg = "
                f"{tre_before.mean():.3f} ± {tre_before.std():.3f} / "
                f"{tre_after_affine.mean():.3f} ± {tre_after_affine.std():.3f} / "
                f"{tre_after_varreg.mean():.3f} ± {tre_after_varreg.std():.3f} / "
            )
