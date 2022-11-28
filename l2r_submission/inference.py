from __future__ import annotations

import logging
import warnings

import numpy as np

from vroc.common_types import PathLike
from vroc.logger import init_fancy_logging

logger = logging.getLogger(__name__)


import SimpleITK as sitk


def load_image_data(
    moving_image_filepath: PathLike,
    fixed_image_filepath: PathLike,
    moving_mask_filepath: PathLike | None = None,
    fixed_mask_filepath: PathLike | None = None,
    moving_labels_filepath: PathLike | None = None,
    fixed_labels_filepath: PathLike | None = None,
) -> dict:
    logger.info(f"Try to load moving image from {moving_image_filepath}")
    logger.info(f"Try to load fixed image from {fixed_image_filepath}")
    logger.info(f"Try to load moving mask from {moving_mask_filepath}")
    logger.info(f"Try to load fixed mask from {fixed_mask_filepath}")
    logger.info(f"Try to load moving from {moving_labels_filepath}")
    logger.info(f"Try to load fixed labels from {fixed_labels_filepath}")

    def try_to_load_image(filepath: PathLike, dtype=None) -> np.ndarray | None:
        if filepath and Path(filepath).exists():
            image = sitk.ReadImage(str(filepath))
            image = sitk.GetArrayFromImage(image)
            image = np.swapaxes(image, 0, 2)
            if dtype:
                image = image.astype(dtype)
        else:
            image = None
            if filepath:
                logger.warning(f"Not found {filepath}")

        return image

    moving_image = sitk.ReadImage(str(moving_image_filepath))
    fixed_image = sitk.ReadImage(str(fixed_image_filepath))
    image_spacing = fixed_image.GetSpacing()[::-1]
    moving_image = sitk.GetArrayFromImage(moving_image)
    fixed_image = sitk.GetArrayFromImage(fixed_image)
    moving_image = np.swapaxes(moving_image, 0, 2)
    fixed_image = np.swapaxes(fixed_image, 0, 2)

    moving_mask = try_to_load_image(moving_mask_filepath, dtype=bool)
    fixed_mask = try_to_load_image(fixed_mask_filepath, dtype=bool)

    if moving_mask is None:
        moving_mask = np.ones_like(moving_image, dtype=bool)
        logger.info(f"Created all-ones moving mask")
    if fixed_mask is None:
        fixed_mask = np.ones_like(fixed_image, dtype=bool)
        logger.info(f"Created all-ones fixed mask")

    moving_labels = try_to_load_image(moving_labels_filepath, dtype=np.uint8)
    fixed_labels = try_to_load_image(fixed_labels_filepath, dtype=np.uint8)

    return dict(
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_mask=moving_mask,
        fixed_mask=fixed_mask,
        moving_labels=moving_labels,
        fixed_labels=fixed_labels,
        image_spacing=image_spacing,
    )


def write_vector_field(vector_field: np.ndarray, output_filepath: PathLike):
    vector_field = np.swapaxes(vector_field, 1, 3)
    vector_field = sitk.GetImageFromArray(vector_field, isVector=False)

    sitk.WriteImage(vector_field, str(output_filepath))

    logger.info(f"Saved vector field to {output_filepath}")


if __name__ == "__main__":
    import json
    import os
    import sys
    from pathlib import Path

    import torch
    from torch.optim import Adam

    from vroc.helper import compute_tre_numpy
    from vroc.keypoints import extract_keypoints
    from vroc.loss import mse_loss
    from vroc.models import DemonsVectorFieldBooster
    from vroc.registration import VrocRegistration

    init_fancy_logging(
        handlers=[logging.StreamHandler(), logging.FileHandler("output/debug.log")]
    )

    logger.setLevel(logging.DEBUG)
    logging.getLogger("vroc").setLevel(logging.DEBUG)
    logging.getLogger("vroc.affine").setLevel(logging.DEBUG)
    logging.getLogger("vroc.models.VarReg").setLevel(logging.DEBUG)

    if len(sys.argv) == 3:
        _, dataset_filename, gpu_id = sys.argv

    dataset_filename = Path(dataset_filename)

    # print(f"Arguments count: {len(sys.argv)}")
    # for data, arg in enumerate(sys.argv):
    #     if data == 1:
    #         print(f"dataset_filename {data:>6}: {arg}")
    #         dataset_filename = str(arg)
    #     elif data == 2:
    #         print(f"GPU ID {data:>6}: {arg}")
    #         gpu_id = int(arg)
    #     else:
    #         print(f" argument {data:>6}: {arg}")

    # for testing without docker
    # gpu_id = 1
    # dataset_filename = Path("/datalake/learn2reg/NLST_testdata/NLST_dataset.json")
    # dataset_filename = Path("/datalake/learn2reg/AbdomenMRMR/AbdomenMRMR_dataset.json")
    # dataset_filename = Path('/datalake/learn2reg/LungCT/LungCT_dataset.json')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name()})")

    data_path = dataset_filename.parent

    with open(dataset_filename) as f:
        dataset_info = json.load(f)

    logger.info(f"Got {dataset_info=}")

    task_name = dataset_info["name"]

    # Check for output_folder
    # out_folder = Path("/datalake/learn2reg/submission_output") / task_name
    out_folder = Path("output") / task_name
    out_folder.mkdir(parents=True, exist_ok=True)

    params = {
        "iterations": 800,
        "tau": 2.25,
        "sigma_x": 1.25,
        "sigma_y": 1.25,
        "sigma_z": 1.25,
        "n_levels": 3,
    }

    registration = VrocRegistration(
        roi_segmenter=None,
        device=device,
    )

    # selection of config
    config = {}

    modalities = list(set(dataset_info["modality"].values()))
    pairing = dataset_info["pairings"]

    logger.info(f"Config: {task_name=}, {modalities=}, {pairing=}")

    if pairing == "paired":
        config.update(
            {
                "keypoint_loss_weight": 1.0,
                "label_loss_weight": 0.0,
                "smoothness_loss_weight": 0.5,
                "image_loss_weight": 0.0,
            }
        )
    elif pairing == "unpaired":
        config.update(
            {
                "keypoint_loss_weight": 0.0,
                "label_loss_weight": 1.0,
                "smoothness_loss_weight": 0.5,
                "image_loss_weight": 0.5,
            }
        )

    logger.info(f"Using task config: {config}")

    for data in dataset_info["registration_test"]:

        logger.info(f"Handle: {data}")

        moving_image_filepath = data_path / data["moving"]
        fixed_image_filepath = data_path / data["fixed"]
        moving_mask_filepath = data_path / data["moving"].replace("images", "masks")
        fixed_mask_filepath = data_path / data["fixed"].replace("images", "masks")
        moving_labels_filepath = data_path / data["moving"].replace(
            "images", "predictedlabels"
        )
        fixed_labels_filepath = data_path / data["fixed"].replace(
            "images", "predictedlabels"
        )

        # load image data
        image_data = load_image_data(
            moving_image_filepath=moving_image_filepath,
            fixed_image_filepath=fixed_image_filepath,
            moving_mask_filepath=moving_mask_filepath,
            fixed_mask_filepath=fixed_mask_filepath,
            moving_labels_filepath=moving_labels_filepath,
            fixed_labels_filepath=fixed_labels_filepath,
        )

        labels = dataset_info.get("labels")
        if labels:
            logger.info(f"Found the following labels: {labels}")
            labels = labels["0"]
            n_label_classes = len(labels)
            logger.info(f"Using {n_label_classes=}")
        else:
            n_label_classes = None

        # extract keypoints if paired images
        if pairing == "paired":
            logger.info("Extracting keypoints")
            with warnings.catch_warnings():
                moving_keypoints, fixed_keypoints = extract_keypoints(
                    moving_image=torch.as_tensor(
                        image_data["moving_image"][None, None], device=device
                    ),
                    fixed_image=torch.as_tensor(
                        image_data["fixed_image"][None, None], device=device
                    ),
                    fixed_mask=torch.as_tensor(
                        image_data["fixed_mask"][None, None], device=device
                    ),
                )

            tre_before = compute_tre_numpy(
                moving_landmarks=moving_keypoints.cpu().detach().numpy()[0],
                fixed_landmarks=fixed_keypoints.cpu().detach().numpy()[0],
                vector_field=None,
                image_spacing=image_data["image_spacing"],
            )

            logger.info(f"Found keypoints: {moving_keypoints.shape}")
        else:
            moving_keypoints, fixed_keypoints = None, None

        # perform registration + boosting
        model = DemonsVectorFieldBooster(n_iterations=4).to(device)
        optimizer = Adam(model.parameters(), lr=2.5e-4)

        valid_value_range = (
            min(image_data["moving_image"].min(), image_data["fixed_image"].min()),
            max(image_data["moving_image"].max(), image_data["fixed_image"].max()),
        )

        logger.info(f"Using {valid_value_range=}")

        if np.sum(image_data["moving_mask"]) * 0.95 > np.sum(image_data["fixed_mask"]):
            gradient_type = "active"
        elif np.sum(image_data["fixed_mask"]) * 0.95 > np.sum(
            image_data["moving_mask"]
        ):
            gradient_type = "passive"
        else:
            gradient_type = "dual"

        logger.info(f"Using {gradient_type=}")

        registration_result = registration.register_and_train_boosting(
            # boosting specific kwargs
            model=model,
            optimizer=optimizer,
            n_iterations=200,
            moving_keypoints=moving_keypoints,
            fixed_keypoints=fixed_keypoints,
            moving_labels=image_data["moving_labels"],
            fixed_labels=image_data["fixed_labels"],
            n_label_classes=n_label_classes,
            image_loss_function="mse",
            # loss weights
            keypoint_loss_weight=config["keypoint_loss_weight"],
            label_loss_weight=config["label_loss_weight"],
            smoothness_loss_weight=config["smoothness_loss_weight"],
            image_loss_weight=config["image_loss_weight"],
            # registration specific kwargs
            moving_image=image_data["moving_image"],
            fixed_image=image_data["fixed_image"],
            moving_mask=image_data["moving_mask"],
            fixed_mask=image_data["fixed_mask"],
            use_masks=True,
            image_spacing=image_data["image_spacing"],
            register_affine=True,
            affine_loss_function=mse_loss,
            affine_step_size=0.01,
            affine_iterations=300,
            force_type="demons",
            gradient_type=gradient_type,
            valid_value_range=valid_value_range,
            early_stopping_delta=0.00001,
            early_stopping_window=None,
            default_parameters=params,
            debug=False,
            return_as_tensor=False,
        )

        vector_field = registration_result.composed_vector_field

        if pairing == "paired":
            tre_after = compute_tre_numpy(
                moving_landmarks=moving_keypoints.cpu().detach().numpy()[0],
                fixed_landmarks=fixed_keypoints.cpu().detach().numpy()[0],
                vector_field=vector_field,
                image_spacing=image_data["image_spacing"],
            )

            logger.info(
                f"Registration results: TRE before/after: "
                f"{tre_before.mean():.3f}/{tre_after.mean():.3f}"
            )

        # naming convention: disp_FIXED_MOVING.nii.gz
        moving_image_id = int(data["moving"].split("_")[1])
        fixed_image_id = int(data["fixed"].split("_")[1])
        output_filepath = (
            out_folder / f"disp_{fixed_image_id:04d}_{moving_image_id:04d}.nii.gz"
        )

        write_vector_field(
            vector_field=vector_field,
            output_filepath=output_filepath,
        )
