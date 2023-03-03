from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Tuple

import lz4.frame
import numpy as np

np.random.seed(1337)
import uuid

import torch
import yaml

from vroc.dataset import Lung4DCTRegistrationDataset
from vroc.helper import (
    compute_tre_numpy,
    get_bounding_box,
    get_robust_bounding_box_3d,
    pad_bounding_box_to_pow_2,
)
from vroc.logger import init_fancy_logging
from vroc.loss import ncc_loss
from vroc.registration import VrocRegistration


def filter_by_metadata(patient_folder: Path, filter_func):
    try:
        with open(patient_folder / "metadata.yml", "rt") as f:
            meta = yaml.safe_load(f)
        filter_result = filter_func(meta)
    except FileNotFoundError:
        logger.warning(f"No metadata.yml found in patient folder {patient_folder}")
        filter_result = False

    return filter_result


def has_no_artifacts(meta: dict) -> bool:
    if (
        meta["artifactness_interpolation"] == 0
        and meta["artifactness_double_structure"] == 0
    ):
        return True
    else:
        return False


def generate_artifact_mask(
    roi_mask: np.ndarray, roi_z_range: Tuple[float, float], artifact_size: int = 1
) -> np.ndarray:
    # calculate valid range along z axis
    roi_bbox = get_robust_bounding_box_3d(roi_mask)

    n_slices = roi_bbox[-1].stop - roi_bbox[-1].start
    start = roi_bbox[-1].start + roi_z_range[0] * n_slices
    stop = roi_bbox[-1].start + roi_z_range[1] * n_slices

    start_slice_range = (
        start,
        max(stop - artifact_size, start + 1),  # stop is at least start + 1
    )

    random_start_slice = np.random.randint(*start_slice_range)

    artifact_mask = np.zeros_like(roi_mask, dtype=bool)
    artifact_mask[..., random_start_slice : random_start_slice + artifact_size] = True

    logger.debug(
        f"Generated artifact mask at "
        f"[{random_start_slice}:{random_start_slice + artifact_size}] "
        f"({artifact_size} slices)"
    )

    return artifact_mask


if __name__ == "__main__":
    init_fancy_logging()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.getLogger("vroc").setLevel(logging.INFO)

    DEVICE = torch.device("cuda:1")
    OUTPUT_FOLDER = Path("/datalake2/vroc_artifact_boosting/v2")

    patients = sorted(
        list(
            Path("/datalake_fast/4d_ct_lung_uke_artifact_free").glob(
                "*_Lunge_amplitudebased_complete"
            )
        )
    )

    patients_no_artifacts = [
        patient
        for patient in patients
        if filter_by_metadata(patient, filter_func=has_no_artifacts)
    ]

    train_dataset = Lung4DCTRegistrationDataset(
        patient_folders=patients_no_artifacts,
        phases=(0, 5),
        train_size=1.0,
        is_train=True,
        input_value_range=None,
        output_value_range=None,
    )

    for data in train_dataset:

        registration = VrocRegistration(
            roi_segmenter=None,
            feature_extractor=None,
            parameter_guesser=None,
            device=DEVICE,
        )

        # exclude artifact region in fixed mask (fixed image is artifact affected image)
        # random aritfact size; low (inclusive) to high (exclusive)
        for artifact_size in range(5, 31):
            moving_image, moving_image_meta = data["images"][5]
            fixed_image, fixed_image_meta = data["images"][0]

            moving_mask, _ = data["masks"][5]
            fixed_mask, _ = data["masks"][0]

            moving_keypoints, fixed_keypoints = data["keypoints"][(5, 0)]

            logger.info(f"Now at {data['patient']}, {artifact_size=}")

            # calculate pow 2 ROI bbox
            roi_bbox = get_robust_bounding_box_3d(fixed_mask)
            padded_roi_bbox = pad_bounding_box_to_pow_2(roi_bbox)

            fixed_artifact_mask = generate_artifact_mask(
                roi_mask=fixed_mask,  # we need 3D input
                roi_z_range=(0.0, 0.5),
                artifact_size=artifact_size,
            )
            # print(np.mean((fixed_image[fixed_artifact_mask] - moving_image[fixed_artifact_mask]) **2)**0.5)

            fixed_artifact_mask = fixed_artifact_mask[
                None, None
            ]  # add color channel again

            # move data to device
            moving_image = torch.as_tensor(moving_image, device=DEVICE)
            fixed_image = torch.as_tensor(fixed_image, device=DEVICE)
            moving_mask = torch.as_tensor(moving_mask, device=DEVICE, dtype=torch.bool)
            fixed_mask = torch.as_tensor(fixed_mask, device=DEVICE, dtype=torch.bool)
            fixed_artifact_mask = torch.as_tensor(
                fixed_artifact_mask, device=DEVICE, dtype=torch.bool
            )
            fixed_artifact_bbox = get_bounding_box(fixed_artifact_mask)
            # exclude artifact region from image registration
            fixed_mask_without_artifact_region = fixed_mask & (~fixed_artifact_mask)

            params = {
                "iterations": 200,
                "tau": 2.25,
                "tau_level_decay": 0.0,
                "tau_iteration_decay": 0.0,
                "sigma_x": 1.25,
                "sigma_y": 1.25,
                "sigma_z": 1.25,
                "sigma_level_decay": 0.0,
                "sigma_iteration_decay": 0.0,
                "n_levels": 3,
            }

            registration_results = {}

            registration_result = registration.register(
                moving_image=moving_image,
                fixed_image=fixed_image,
                moving_mask=moving_mask,
                fixed_mask=fixed_mask,
                register_affine=False,
                affine_loss_function=ncc_loss,
                force_type="demons",
                gradient_type="passive",
                valid_value_range=(-1024, 3071),
                early_stopping_delta=0.00001,
                early_stopping_window=None,
                debug=False,
                default_parameters=params,
                return_as_tensor=False,
            )
            registration_results["without_artifact"] = registration_result

            aspect = (
                fixed_image_meta["image_spacing"][0]
                / fixed_image_meta["image_spacing"][2]
            )
            clim = (-800, 200)
            imshow_kwargs = {"aspect": aspect, "clim": clim}
            # fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
            # ax[0, 0].imshow(data["moving_image"][0, :, 256, :], **imshow_kwargs)
            # ax[0, 1].imshow(data["fixed_image"][0, :, 256, :], **imshow_kwargs)
            # ax[0, 2].imshow(
            #     registration_result.warped_moving_image[:, 256, :], **imshow_kwargs
            # )
            # ax[1, 0].imshow(registration_result.moving_mask[:, 256, :], aspect=aspect)
            # ax[1, 1].imshow(registration_result.fixed_mask[:, 256, :], aspect=aspect)
            # ax[1, 2].imshow(
            #     registration_result.composed_vector_field[2, :, 256, :], aspect=aspect
            # )

            tre_before = compute_tre_numpy(
                moving_landmarks=moving_keypoints,
                fixed_landmarks=fixed_keypoints,
                vector_field=None,
                image_spacing=fixed_image_meta["image_spacing"],
                fixed_bounding_box=None,
            )
            tre_artifact_before = compute_tre_numpy(
                moving_landmarks=moving_keypoints,
                fixed_landmarks=fixed_keypoints,
                vector_field=None,
                image_spacing=fixed_image_meta["image_spacing"],
                fixed_bounding_box=fixed_artifact_bbox[-3:],  # just spatial bbox
            )

            tre_after = compute_tre_numpy(
                moving_landmarks=moving_keypoints,
                fixed_landmarks=fixed_keypoints,
                vector_field=registration_result.composed_vector_field,
                image_spacing=fixed_image_meta["image_spacing"],
                fixed_bounding_box=None,
            )
            tre_artifact_after = compute_tre_numpy(
                moving_landmarks=moving_keypoints,
                fixed_landmarks=fixed_keypoints,
                vector_field=registration_result.composed_vector_field,
                image_spacing=fixed_image_meta["image_spacing"],
                fixed_bounding_box=fixed_artifact_bbox[-3:],  # just spatial bbox
            )

            logger.debug(
                f"Without artifact: {tre_before.mean()=:.3f} / {tre_after.mean()=:.3f} // {tre_artifact_before.mean()=:.3f} / {tre_artifact_after.mean()=:.3f}"
            )

            registration_result = registration.register(
                moving_image=moving_image,
                fixed_image=fixed_image,
                moving_mask=moving_mask,
                fixed_mask=fixed_mask_without_artifact_region,
                register_affine=False,
                affine_loss_function=ncc_loss,
                force_type="demons",
                gradient_type="passive",
                valid_value_range=(-1024, 3071),
                early_stopping_delta=0.00001,
                early_stopping_window=None,
                debug=False,
                default_parameters=params,
                return_as_tensor=False,
            )
            registration_results["with_artifact"] = registration_result

            # fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
            # ax[0, 0].imshow(data["moving_image"][0, :, 256, :], **imshow_kwargs)
            # ax[0, 1].imshow(data["fixed_image"][0, :, 256, :], **imshow_kwargs)
            # ax[0, 2].imshow(
            #     registration_result.warped_moving_image[:, 256, :], **imshow_kwargs
            # )
            # ax[1, 0].imshow(registration_result.moving_mask[:, 256, :], aspect=aspect)
            # ax[1, 1].imshow(registration_result.fixed_mask[:, 256, :], aspect=aspect)
            # ax[1, 2].imshow(
            #     registration_result.composed_vector_field[2, :, 256, :], aspect=aspect
            # )

            tre_before = compute_tre_numpy(
                moving_landmarks=moving_keypoints,
                fixed_landmarks=fixed_keypoints,
                vector_field=None,
                image_spacing=fixed_image_meta["image_spacing"],
                fixed_bounding_box=None,
            )
            tre_artifact_before = compute_tre_numpy(
                moving_landmarks=moving_keypoints,
                fixed_landmarks=fixed_keypoints,
                vector_field=None,
                image_spacing=fixed_image_meta["image_spacing"],
                fixed_bounding_box=fixed_artifact_bbox[-3:],  # just spatial bbox
            )

            tre_after = compute_tre_numpy(
                moving_landmarks=moving_keypoints,
                fixed_landmarks=fixed_keypoints,
                vector_field=registration_result.composed_vector_field,
                image_spacing=fixed_image_meta["image_spacing"],
                fixed_bounding_box=None,
            )
            tre_artifact_after = compute_tre_numpy(
                moving_landmarks=moving_keypoints,
                fixed_landmarks=fixed_keypoints,
                vector_field=registration_result.composed_vector_field,
                image_spacing=fixed_image_meta["image_spacing"],
                fixed_bounding_box=fixed_artifact_bbox[-3:],  # just spatial bbox
            )

            logger.debug(
                f"With artifact: {tre_before.mean()=:.3f} / {tre_after.mean()=:.3f} // {tre_artifact_before.mean()=:.3f} / {tre_artifact_after.mean()=:.3f}"
            )

            image_data_filepath = (
                OUTPUT_FOLDER / Path(data["patient"]).name / "image_data.pkl.lz4"
            )
            registration_data_filepath = (
                OUTPUT_FOLDER
                / Path(data["patient"]).name
                / f"registration_data_size_{artifact_size:02d}_{uuid.uuid4().hex[:8]}.pkl.lz4"
            )

            roi_slicing = (..., *padded_roi_bbox)

            if not image_data_filepath.exists():
                image_data_filepath.parent.mkdir(exist_ok=True)
                image_data = dict(
                    patient=data["patient"],
                    moving_image_name=moving_image_meta["filepath"],
                    fixed_image_name=fixed_image_meta["filepath"],
                    moving_image=np.asarray(
                        registration_results["without_artifact"].moving_image,
                        dtype=np.int16,
                    ),
                    fixed_image=np.asarray(
                        registration_results["without_artifact"].fixed_image,
                        dtype=np.int16,
                    ),
                    moving_mask=np.asarray(
                        registration_results["without_artifact"].moving_mask, dtype=bool
                    ),
                    fixed_mask=np.asarray(
                        registration_results["without_artifact"].fixed_mask, dtype=bool
                    ),
                    fixed_bounding_box=roi_bbox,
                    padded_fixed_bounding_box=padded_roi_bbox,
                    moving_keypoints=moving_keypoints,
                    fixed_keypoints=fixed_keypoints,
                    tre_before=tre_before,
                    image_spacing=fixed_image_meta["image_spacing"],
                )
                with lz4.frame.open(image_data_filepath, "wb") as f:
                    pickle.dump(image_data, f)

            registration_data = dict(
                patient=data["patient"],
                moving_image_name=moving_image_meta["filepath"],
                fixed_image_name=fixed_image_meta["filepath"],
                artifact_size=artifact_size,
                fixed_artifact_mask=np.asarray(
                    fixed_artifact_mask[0, 0].detach().cpu().numpy(), dtype=bool
                ),
                fixed_artifact_bbox=fixed_artifact_bbox[-3:],
                registration_parameters=params,
                vector_field_without_artifact=registration_results[
                    "without_artifact"
                ].composed_vector_field,
                vector_field_with_artifact=registration_results[
                    "with_artifact"
                ].composed_vector_field,
                tre_after=tre_after,
                tre_artifact_before=tre_artifact_before,
                tre_artifact_after=tre_artifact_after,
            )
            with lz4.frame.open(registration_data_filepath, "wb") as f:
                pickle.dump(registration_data, f)
