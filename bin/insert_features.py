import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from vroc.feature_extractor import OrientedHistogramFeatureExtrator
from vroc.hyperopt_database.client import DatabaseClient
from vroc.logger import LogFormatter

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(LogFormatter())
logging.basicConfig(handlers=[handler])

logging.getLogger(__name__).setLevel(logging.DEBUG)
logging.getLogger("vroc").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

ROOT_DIR = Path("/datalake/learn2reg/NLST")
OVERWRITE = True

client = DatabaseClient("/datalake/learn2reg/param_sampling_v2.sqlite")

feature_extractor = OrientedHistogramFeatureExtrator(n_bins=16, device="cuda:0")

image_pairs = client.fetch_image_pairs()

for image_pair in image_pairs:
    moving_image = image_pair["moving_image"]
    fixed_image = image_pair["fixed_image"]

    moving_image_filepath = ROOT_DIR / moving_image.name
    fixed_image_filepath = ROOT_DIR / fixed_image.name
    moving_mask_filepath = ROOT_DIR / moving_image.name.replace("images", "masks")
    fixed_mask_filepath = ROOT_DIR / fixed_image.name.replace("images", "masks")

    logger.info(f"{moving_image_filepath} / {fixed_image_filepath}")

    moving_image_arr = sitk.ReadImage(str(moving_image_filepath))
    fixed_image_arr = sitk.ReadImage(str(fixed_image_filepath))
    moving_mask_arr = sitk.ReadImage(str(moving_mask_filepath))
    fixed_mask_arr = sitk.ReadImage(str(fixed_mask_filepath))

    image_spacing = fixed_image_arr.GetSpacing()[::-1]

    moving_image_arr = sitk.GetArrayFromImage(moving_image_arr)
    fixed_image_arr = sitk.GetArrayFromImage(fixed_image_arr)
    moving_mask_arr = sitk.GetArrayFromImage(moving_mask_arr)
    fixed_mask_arr = sitk.GetArrayFromImage(fixed_mask_arr)

    moving_image_arr = np.swapaxes(moving_image_arr, 0, 2)
    fixed_image_arr = np.swapaxes(fixed_image_arr, 0, 2)
    moving_mask_arr = np.swapaxes(moving_mask_arr, 0, 2)
    fixed_mask_arr = np.swapaxes(fixed_mask_arr, 0, 2)

    feature = feature_extractor.extract(
        fixed_image=fixed_image_arr,
        moving_image=moving_image_arr,
        fixed_mask=fixed_mask_arr,
        moving_mask=moving_mask_arr,
        image_spacing=image_spacing,
    )

    client.insert_image_pair_feature(
        moving_image=moving_image,
        fixed_image=fixed_image,
        feature_name=feature_extractor.feature_name,
        feature=feature,
        overwrite=True,
    )
