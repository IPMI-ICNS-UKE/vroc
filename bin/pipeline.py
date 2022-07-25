from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from vroc.dataset import BaseDataset
from vroc.feature_extractor import FeatureExtractor, calculate_oriented_histogram
from vroc.models import TrainableVarRegBlock, UNet
from vroc.preprocessing import affine_registration, resample_image_spacing
from vroc.segmentation import LungSegmenter2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_fixed_filepath = ""
image_moving_filepath = ""
mask_fixed_filepath = ""
mask_moving_filepath = ""

spacing = (1, 1, 1)

image_fixed = BaseDataset.load_and_preprocess(image_fixed_filepath)
image_moving = BaseDataset.load_and_preprocess(image_moving_filepath)

if mask_fixed_filepath and mask_moving_filepath:
    mask_fixed = BaseDataset.load_and_preprocess(mask_fixed_filepath)
    mask_moving = BaseDataset.load_and_preprocess(mask_moving_filepath)
else:
    mask_fixed = None
    mask_moving = None

image_fixed = resample_image_spacing(image=image_fixed, new_spacing=spacing)
image_moving = resample_image_spacing(image=image_moving, new_spacing=spacing)

# TODO: check affine registration performance
image_warped, affine_transform = affine_registration(image_fixed, image_moving)

# TODO: scale intensity range

image_array_fixed = sitk.GetArrayFromImage(image_fixed)
image_array_warped = sitk.GetArrayFromImage(image_warped)

image_array_fixed = np.swapaxes(image_array_fixed, 0, 2)
image_array_warped = np.swapaxes(image_array_warped, 0, 2)

if mask_fixed_filepath and mask_moving_filepath:
    mask_fixed = resample_image_spacing(image=mask_fixed, new_spacing=spacing)
    mask_array_fixed = sitk.GetArrayFromImage(mask_fixed)
    mask_array_fixed = np.swapaxes(mask_array_fixed, 0, 2)

    mask_moving = resample_image_spacing(image=mask_moving, new_spacing=spacing)
    mask_array_moving = sitk.GetArrayFromImage(mask_moving)
    mask_array_moving = np.swapaxes(mask_array_moving, 0, 2)
else:
    lung_segmenter = LungSegmenter2D(
        model=UNet().to("cuda"),
        state_filepath=Path(""),
    )
    mask_array_fixed = lung_segmenter.segment(image_array_fixed)
    mask_array_moving = lung_segmenter.segment(image_array_warped)

feature_extractor = FeatureExtractor(state_filepath="")

demons_features = calculate_oriented_histogram(
    image_array_fixed,
    image_array_warped,
    mask_array_fixed,
    mask_array_moving,
    image_spacing=spacing,
)
ae_features_fixed = feature_extractor.infer(image_array_fixed)
ae_features_warped = feature_extractor.infer(image_array_warped)

fixed = torch.as_tensor(image_array_fixed, device=device)[None, None]
moving = torch.as_tensor(image_array_warped, device=device)[None, None]
mask = torch.as_tensor(mask_array_fixed, device=device)[None, None]

# TODO: match new features with hyperopt_database features and get corresponding demons hyperparams
params = {}

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

warped, demons_transform, _ = varreg.forward(fixed, mask, moving, spacing)

total_transform = sitk.CompositeTransform([affine_transform, demons_transform])
