from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import nevergrad as ng
import numpy as np
import typer
from scipy.ndimage import binary_dilation

import vroc.database.models as orm
from vroc.dataset import NLSTDataset
from vroc.decay import half_life_to_lambda
from vroc.helper import compute_tre_numpy
from vroc.hyperopt_database.client import DatabaseClient
from vroc.logger import LogFormatter
from vroc.loss import mse_loss, ncc_loss
from vroc.metrics import root_mean_squared_error
from vroc.registration import VrocRegistration

logger = logging.getLogger(__name__)


def setup_optimizer(
    moving_image: orm.Image,
    fixed_image: orm.Image,
    database: DatabaseClient,
    metric: orm.Metric,
    optimizer_name: str = "RandomSearch",
):
    init_params = VrocRegistration.DEFAULT_REGISTRATION_PARAMETERS

    param_config = ng.p.Instrumentation(
        iterations=ng.p.Constant(800),  # handled by early stooping
        tau=ng.p.Scalar(lower=0.5, upper=10.0, init=init_params["tau"]),
        tau_level_decay=ng.p.Scalar(
            lower=-half_life_to_lambda(4),
            upper=half_life_to_lambda(4),
            init=init_params["tau_level_decay"],
        ),
        tau_iteration_decay=ng.p.Scalar(
            lower=-half_life_to_lambda(800),
            upper=half_life_to_lambda(800),
            init=init_params["tau_iteration_decay"],
        ),
        sigma_x=ng.p.Scalar(lower=0.5, upper=5.0, init=init_params["sigma_x"]),
        sigma_y=ng.p.Scalar(lower=0.5, upper=5.0, init=init_params["sigma_y"]),
        sigma_z=ng.p.Scalar(lower=0.5, upper=5.0, init=init_params["sigma_z"]),
        n_levels=ng.p.Scalar(
            lower=1, upper=4, init=init_params["n_levels"]
        ).set_integer_casting(),
        sigma_level_decay=ng.p.Scalar(
            lower=-half_life_to_lambda(4),
            upper=half_life_to_lambda(4),
            init=init_params["sigma_level_decay"],
        ),
        sigma_iteration_decay=ng.p.Scalar(
            lower=-half_life_to_lambda(800),
            upper=half_life_to_lambda(800),
            init=init_params["sigma_iteration_decay"],
        ),
    )

    optimizer_class = ng.optimizers.registry[optimizer_name]
    optimizer = optimizer_class(parametrization=param_config)

    previous_runs = database.fetch_runs(
        moving_image=moving_image, fixed_image=fixed_image, metric=metric
    )

    logging.info(
        f"Pretrain optimizer of {moving_image.name}/{fixed_image.name} on "
        f"{len(previous_runs)} previous runs"
    )
    for previous_run in previous_runs:
        # this is length 1 list as we only select one metric
        run_metrics = previous_run["run_metrics"]
        run_metric = run_metrics[0]

        loss = run_metric["value_after"]

        optimizer.suggest(**previous_run["parameters"])
        params = optimizer.ask()
        optimizer.tell(params, loss)

    return optimizer


def sample_parameter_space(
    nlst_dir: Path,
    database_filepath: Path,
    iterations: int = -1,
    iterations_per_image: int = 100,
    optimizer_name: str = "TwoPointsDE",
    loss_name: str = "TRE_MEAN",
    device: str = "cuda:0",
    i_worker: Optional[int] = None,
    n_worker: Optional[int] = None,
    log_level: str = "INFO",
):
    log_level = getattr(logging, log_level.upper())

    train_dataset = NLSTDataset(
        nlst_dir, i_worker=i_worker, n_worker=n_worker, dilate_masks=1
    )

    # some constants for the database
    DATASET = "NLST"
    ANATOMY = "LUNG"
    MODALITY = "CT"

    database = DatabaseClient(database_filepath)

    METRICS = {
        "TRE_MEAN": database.get_or_create_metric(
            name="TRE_MEAN", lower_is_better=True
        ),
        "TRE_STD": database.get_or_create_metric(name="TRE_STD", lower_is_better=True),
        "RMSE": database.get_or_create_metric(name="RMSE", lower_is_better=True),
    }

    iterations_done = 0
    while True:
        for data in train_dataset:
            # insert moving and fixed image into database
            moving_image_db = database.get_or_create_image(
                image_name=data["moving_image_name"],
                modality=MODALITY,
                anatomy=ANATOMY,
                dataset=DATASET,
            )
            fixed_image_db = database.get_or_create_image(
                image_name=data["fixed_image_name"],
                modality=MODALITY,
                anatomy=ANATOMY,
                dataset=DATASET,
            )

            # get images/masks/... and remove color channel
            moving_image = data["moving_image"][0]
            fixed_image = data["fixed_image"][0]
            moving_mask = data["moving_mask"][0]
            fixed_mask = data["fixed_mask"][0]
            moving_keypoints = data["moving_keypoints"]
            fixed_keypoints = data["fixed_keypoints"]
            image_spacing = data["image_spacing"]

            moving_mask = binary_dilation(
                moving_mask.astype(np.uint8), iterations=1
            ).astype(np.uint8)
            fixed_mask = binary_dilation(
                fixed_mask.astype(np.uint8), iterations=1
            ).astype(np.uint8)

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

            optimizer = setup_optimizer(
                moving_image=moving_image_db,
                fixed_image=fixed_image_db,
                database=database,
                metric=METRICS[loss_name],
                optimizer_name=optimizer_name,
            )
            for _ in range(iterations_per_image):
                params = optimizer.ask()

                registration = VrocRegistration(
                    roi_segmenter=None,
                    feature_extractor=None,
                    parameter_guesser=None,
                    device=device,
                )
                registration_result = registration.register(
                    moving_image=moving_image,
                    fixed_image=fixed_image,
                    moving_mask=moving_mask,
                    fixed_mask=fixed_mask,
                    register_affine=True,
                    affine_loss_fn=ncc_loss,
                    force_type="demons",
                    gradient_type="active",
                    valid_value_range=(-1024, 3071),
                    early_stopping_delta=0.00001,
                    early_stopping_window=400,
                    default_parameters=params.kwargs,
                    debug=False,
                )
                vector_field = registration_result.composed_vector_field
                warped_image = registration_result.warped_moving_image

                # calculate TRE after registration

                tre_after = compute_tre_numpy(
                    moving_landmarks=moving_keypoints,
                    fixed_landmarks=fixed_keypoints,
                    vector_field=vector_field,
                    image_spacing=image_spacing,
                )
                tre_after_mean = tre_after.mean()
                tre_after_std = tre_after.std()

                rmse_before = root_mean_squared_error(
                    moving_image, fixed_image, mask=fixed_mask
                )
                rmse_after = root_mean_squared_error(
                    warped_image, fixed_image, mask=fixed_mask
                )

                run = database.insert_run(
                    moving_image=moving_image_db,
                    fixed_image=fixed_image_db,
                    parameters=params.kwargs,
                )
                database.insert_run_metric(
                    run=run,
                    metric=METRICS["TRE_MEAN"],
                    value_before=tre_before_mean,
                    value_after=tre_after_mean,
                )
                database.insert_run_metric(
                    run=run,
                    metric=METRICS["TRE_STD"],
                    value_before=tre_before_std,
                    value_after=tre_after_std,
                )
                database.insert_run_metric(
                    run=run,
                    metric=METRICS["RMSE"],
                    value_before=rmse_before,
                    value_after=rmse_after,
                )

                optimizer.tell(params, loss=tre_after_mean)

                # format floats inside dict for printing
                pretty_params = {
                    param_name: round(param_value, 2)
                    if isinstance(param_value, float)
                    else param_value
                    for (param_name, param_value) in params.kwargs.items()
                }

                logger.info(
                    f"Registration of {moving_image_db.name}/{fixed_image_db.name} "
                    f"with params {pretty_params}: "
                    f"TRE: before = {tre_before_mean:.3f} ± {tre_before_std:.3f} // "
                    f"after = {tre_after_mean:.3f} ± {tre_after_std:.3f}"
                )

        iterations_done += 1
        if 0 < iterations <= iterations_done:
            break


if __name__ == "__main__":
    np.random.seed(1337)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(LogFormatter())
    logging.basicConfig(handlers=[handler])

    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.getLogger("vroc").setLevel(logging.INFO)

    typer.run(sample_parameter_space)

    # I_WORKER = 0
    #
    # sample_parameter_space(
    #     nlst_dir=Path("/datalake/learn2reg/NLST"),
    #     database_filepath=Path("/datalake/learn2reg/param_sampling_test.sqlite"),
    #     optimizer_name="TwoPointsDE",
    #     iterations_per_image=10,
    #     n_worker=1,
    #     i_worker=I_WORKER,
    #     device=f"cuda:{I_WORKER}",
    # )
