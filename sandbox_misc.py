import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

from vroc.dataset import Lung4DRegistrationDataset
from vroc.helper import get_robust_bounding_box_3d

logger = logging.getLogger(__name__)


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

    return artifact_mask


patients = sorted(
    list(
        Path("/datalake_fast/xxx_4DCT_organ_type_status_nii").glob(
            "*_Lunge_amplitudebased_complete"
        )
    )
)

patients_no_artifacts = [
    patient
    for patient in patients
    if filter_by_metadata(patient, filter_func=has_no_artifacts)
]


dataset = Lung4DRegistrationDataset(patient_folders=patients_no_artifacts)

for data in dataset:
    moving_image = data["moving_image"][0]
    fixed_image = data["fixed_image"][0]
    moving_mask = data["moving_mask"][0]
    fixed_mask = data["fixed_mask"][0]

    artifact_mask = generate_artifact_mask(
        roi_mask=fixed_mask, roi_z_range=(0.0, 0.3), artifact_size=10
    )

    roi_bbox = get_robust_bounding_box_3d(fixed_mask, bbox_range=(0.001, 0.999))
    roi_mask = np.zeros_like(fixed_mask, dtype=bool)
    roi_mask[roi_bbox] = True

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    ax[0, 0].imshow(moving_image[:, 256, :])
    ax[0, 1].imshow(fixed_image[:, 256, :])
    ax[0, 2].imshow(roi_mask[:, 256, :])

    ax[1, 0].imshow(moving_mask[:, 256, :])
    ax[1, 1].imshow(fixed_mask[:, 256, :])
    ax[1, 2].imshow(artifact_mask[:, 256, :])

    break
