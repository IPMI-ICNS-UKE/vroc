import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from vroc.database.client import DatabaseClient
from vroc.feature_extractor import OrientedHistogramFeatureExtrator
from vroc.hyperopt_database.client import DatabaseClient as HyperoptDatabaseClient
from vroc.logger import LogFormatter

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(LogFormatter())
logging.basicConfig(handlers=[handler])

logging.getLogger(__name__).setLevel(logging.DEBUG)
logging.getLogger("vroc").setLevel(logging.DEBUG)

hyperopt_client = HyperoptDatabaseClient(
    "/datalake/learn2reg/hyperparams_runs/merged_runs.sqlite"
)
client = DatabaseClient("/datalake/learn2reg/best_parameters.sqlite")

feature_extractor = OrientedHistogramFeatureExtrator(embedder=None, device="cuda:0")

best_runs = hyperopt_client.fetch_best_runs(k_best=1, as_dataframe=False)

DATASET = "NLST_LEARN2REG_TRAIN"
MODALITY = "CT"
ANATOMY = "LUNG"
ROOT_DIR = Path("/datalake/learn2reg/NLST_fixed")

for best_run in best_runs:
    moving_image_name = f"{best_run['image']}_0001"
    fixed_image_name = f"{best_run['image']}_0000"

    logging.info(f"{moving_image_name} / {fixed_image_name}")

    str(ROOT_DIR / "imagesTr" / f"{fixed_image_name}.nii.gz")

    moving_image = sitk.ReadImage(
        str(ROOT_DIR / "imagesTr" / f"{moving_image_name}.nii.gz")
    )
    fixed_image = sitk.ReadImage(
        str(ROOT_DIR / "imagesTr" / f"{fixed_image_name}.nii.gz")
    )
    moving_mask = sitk.ReadImage(
        str(ROOT_DIR / "masksTr" / f"{moving_image_name}.nii.gz")
    )
    fixed_mask = sitk.ReadImage(
        str(ROOT_DIR / "masksTr" / f"{fixed_image_name}.nii.gz")
    )

    image_spacing = fixed_image.GetSpacing()[::-1]

    moving_image = sitk.GetArrayFromImage(moving_image)
    fixed_image = sitk.GetArrayFromImage(fixed_image)
    moving_mask = sitk.GetArrayFromImage(moving_mask)
    fixed_mask = sitk.GetArrayFromImage(fixed_mask)

    moving_image = np.swapaxes(moving_image, 0, 2)
    fixed_image = np.swapaxes(fixed_image, 0, 2)
    moving_mask = np.swapaxes(moving_mask, 0, 2)
    fixed_mask = np.swapaxes(fixed_mask, 0, 2)

    features = feature_extractor.extract(
        fixed_image=fixed_image,
        moving_image=moving_image,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        image_spacing=image_spacing,
    )

    moving_image = client.insert_image(
        image_name=moving_image_name,
        modality=MODALITY,
        anatomy=ANATOMY,
        dataset=DATASET,
    )

    fixed_image = client.insert_image(
        image_name=fixed_image_name, modality=MODALITY, anatomy=ANATOMY, dataset=DATASET
    )

    # inset best parameters
    client.insert_best_parameters(
        moving_image=moving_image,
        fixed_image=fixed_image,
        parameters=best_run["parameters"],
        metric_before=best_run["metric_before"],
        metric_after=best_run["metric_after"],
    )
    client.insert_image_pair_features(
        moving_image=moving_image, fixed_image=fixed_image, features=features
    )
