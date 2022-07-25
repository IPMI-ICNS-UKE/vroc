import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from vroc.common_types import PathLike
from vroc.dataset import NLSTDataset
from vroc.decorators import convert
from vroc.hyperopt_database.client import DatabaseClient
from vroc.metrics import mse_improvement
from vroc.models import TrainableVarRegBlock

param_config = ng.p.Instrumentation(
    iterations=ng.p.Scalar(lower=100, upper=1000).set_integer_casting(),
    tau=ng.p.Scalar(lower=0.5, upper=10.0),
    sigma_x=ng.p.Scalar(lower=0.5, upper=5.0),
    sigma_y=ng.p.Scalar(lower=0.5, upper=5.0),
    sigma_z=ng.p.Scalar(lower=0.5, upper=5.0),
    n_levels=ng.p.Scalar(lower=1, upper=4).set_integer_casting(),
)


def setup_optimizer(
    image_id: str, database_client: DatabaseClient, optimizer_name: str = "RandomSearch"
):
    optimizer_class = ng.optimizers.registry[optimizer_name]
    optimizer = optimizer_class(parametrization=param_config)

    previous_runs = database_client.fetch_runs(image_id)
    logging.info(
        f"Pretrain optimizer of {image_id} on {len(previous_runs)} previous runs"
    )
    for previous_run in previous_runs:
        loss = mse_improvement(
            before=previous_run["metric_before"], after=previous_run["metric_after"]
        )

        optimizer.suggest(**previous_run["parameters"])
        params = optimizer.ask()
        optimizer.tell(params, loss)

    return optimizer


def sample_parameter_space(
    nlst_dir: Path,
    database_filepath: Path,
    iterations: int = -1,
    iterations_per_image: int = 10,
    optimizer_name: str = "TwoPointsDE",
    device: str = "cuda:0",
    i_worker: Optional[int] = None,
    n_worker: Optional[int] = None,
    log_level: str = "INFO",
):
    log_level = getattr(logging, log_level.upper())

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    train_dataset = NLSTDataset(nlst_dir, i_worker=i_worker, n_worker=n_worker)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    database = DatabaseClient(database_filepath)

    iterations_done = 0
    while True:
        for data in dataloader:
            image_id = data["id"][0]
            fixed_image = data["fixed_image"].to(device)
            fixed_mask = data["fixed_mask"].to(device)
            moving_image = data["moving_image"].to(device)
            spacing = data["spacing"][0]

            optimizer = setup_optimizer(
                image_id, database_client=database, optimizer_name=optimizer_name
            )
            for _ in range(iterations_per_image):
                params = optimizer.ask()

                scale_factors = tuple(
                    1 / 2**i_level
                    for i_level in reversed(range(params.kwargs["n_levels"]))
                )
                varreg = TrainableVarRegBlock(
                    iterations=params.kwargs["iterations"],
                    scale_factors=scale_factors,
                    demon_forces="symmetric",
                    tau=params.kwargs["tau"],
                    regularization_sigma=(
                        params.kwargs["sigma_x"],
                        params.kwargs["sigma_y"],
                        params.kwargs["sigma_z"],
                    ),
                    restrict_to_mask=True,
                ).to(device)

                with torch.no_grad():
                    warped_moving_image, vector_field, misc = varreg.forward(
                        fixed_image, fixed_mask, moving_image, spacing
                    )

                database.insert_run(
                    image_id=image_id,
                    parameters=params.kwargs,
                    metric_before=misc["metric_before"],
                    metric_after=misc["metric_after"],
                    level_metrics=misc["level_metrics"],
                )

                loss = (misc["metric_after"] - misc["metric_before"]) / misc[
                    "metric_before"
                ]
                optimizer.tell(params, loss=loss)

                best_run = database.fetch_best_run(image_id)
                best_loss = mse_improvement(
                    before=best_run["metric_before"], after=best_run["metric_after"]
                )

                logger.info(
                    f"Registration of {image_id} "
                    f"with params {params.kwargs}: {loss=:.3f} ({best_loss=:.3f})"
                )

        iterations_done += 1
        if 0 < iterations <= iterations_done:
            break


if __name__ == "__main__":
    typer.run(sample_parameter_space)

    # sample_parameter_space(
    #     nlst_dir=Path("/datalake/learn2reg/NLST"),
    #     database_filepath=Path("/datalake/learn2reg/merged_runs.sqlite"),
    #     optimizer_name="TwoPointsDE",
    #     iterations_per_image=3,
    #     n_worker=6,
    #     i_worker=1,
    # )
