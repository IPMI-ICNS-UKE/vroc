import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from vroc.common_types import PathLike
from vroc.database.client import DatabaseClient
from vroc.dataset import NLSTDataset
from vroc.decorators import convert
from vroc.models import TrainableVarRegBlock

param_config = {
    "iterations": {"min": 100, "max": 1000, "kind": "int"},
    "tau": {"min": 0.5, "max": 10.0, "kind": "float"},
    "sigma_x": {"min": 0.5, "max": 5.0, "kind": "float"},
    "sigma_y": {"min": 0.5, "max": 5.0, "kind": "float"},
    "sigma_z": {"min": 0.5, "max": 5.0, "kind": "float"},
    "n_levels": {"min": 1, "max": 4, "kind": "int"},
}


def sample_parameters(config: dict) -> dict:
    sampled = {}
    for param_name, param_config in config.items():
        kind = param_config["kind"]

        if kind == "int":
            sampled_value = np.random.randint(
                param_config["min"], param_config["max"] + 1
            )
        elif kind == "float":
            sampled_value = np.random.uniform(param_config["min"], param_config["max"])
        else:
            raise NotImplementedError

        sampled[param_name] = sampled_value

    return sampled


def sample_parameter_space(
    nlst_dir: Path,
    database_filepath: Path,
    iterations: int = -1,
    iterations_per_image: int = 10,
    device: str = "cuda:0",
    log_level: str = "INFO",
):
    log_level = getattr(logging, log_level.upper())

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    train_dataset = NLSTDataset(nlst_dir)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    database = DatabaseClient(database_filepath)

    iterations_done = 0
    while True:
        for data in dataloader:
            for _ in range(iterations_per_image):
                params = sample_parameters(param_config)

                logger.info(f"Started registration with params {params}")

                scale_factors = tuple(
                    1 / 2**i_level for i_level in reversed(range(params["n_levels"]))
                )
                varreg = TrainableVarRegBlock(
                    iterations=params["iterations"],
                    scale_factors=scale_factors,
                    demon_forces="symmetric",
                    tau=params["tau"],
                    regularization_sigma=(
                        params["sigma_x"],
                        params["sigma_y"],
                        params["sigma_z"],
                    ),
                    restrict_to_mask=True,
                ).to(device)

                image_id = data["id"][0]
                fixed_image = data["fixed_image"].to(device)
                fixed_mask = data["fixed_mask"].to(device)
                moving_image = data["moving_image"].to(device)
                spacing = data["spacing"][0]

                with torch.no_grad():
                    warped_moving_image, vector_field, misc = varreg.forward(
                        fixed_image, fixed_mask, moving_image, spacing
                    )

                database.insert_run(
                    image_id=image_id,
                    parameters=params,
                    metric_before=misc["metric_before"],
                    metric_after=misc["metric_after"],
                    level_metrics=misc["level_metrics"],
                )
        iterations_done += 1
        if 0 < iterations <= iterations_done:
            break


if __name__ == "__main__":
    typer.run(sample_parameter_space)

    # sample_parameter_space(
    #     nlst_dir=Path('/datalake/learn2reg/NLST'),
    #     database_filepath=Path('/datalake/learn2reg/runs.sqlite')
    # )
