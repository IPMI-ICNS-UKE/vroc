from __future__ import annotations

import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def resample_image_spacing(
    image: sitk.Image,
    new_spacing: Tuple[float, float, float],
    resampler=sitk.sitkLinear,
    default_voxel_value=0.0,
):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2]))),
    ]
    resampled_img = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        resampler,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        default_voxel_value,
        image.GetPixelID(),
    )
    return resampled_img


def resample_image_size(
    image: sitk.Image,
    new_size: Tuple[int, int, int],
    resampler=sitk.sitkLinear,
    default_voxel_value=0.0,
):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_spacing = [
        original_size[0] * (original_spacing[0] / new_size[0]),
        original_size[1] * (original_spacing[1] / new_size[1]),
        original_size[2] * (original_spacing[2] / new_size[2]),
    ]
    resampled_img = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        resampler,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        default_voxel_value,
        image.GetPixelID(),
    )
    return resampled_img


def robust_bounding_box_3d(
    image: np.ndarray, bbox_range: Tuple[float, float] = (0.01, 0.99), padding: int = 0
) -> Tuple[slice, slice, slice]:
    """
    image : mask

    """

    x = np.cumsum(image.sum(axis=(1, 2)))
    y = np.cumsum(image.sum(axis=(0, 2)))
    z = np.cumsum(image.sum(axis=(0, 1)))

    x = x / x[-1]
    y = y / y[-1]
    z = z / z[-1]

    x_min, x_max = np.searchsorted(x, bbox_range[0]), np.searchsorted(x, bbox_range[1])
    y_min, y_max = np.searchsorted(y, bbox_range[0]), np.searchsorted(y, bbox_range[1])
    z_min, z_max = np.searchsorted(z, bbox_range[0]), np.searchsorted(z, bbox_range[1])

    x_min, x_max = max(x_min - padding, 0), min(x_max + padding, image.shape[0])
    y_min, y_max = max(y_min - padding, 0), min(y_max + padding, image.shape[1])
    z_min, z_max = max(z_min - padding, 0), min(z_max + padding, image.shape[2])

    return np.index_exp[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]


def crop_background(img: sitk.Image, print_summary=False) -> sitk.Image:
    """Crop background based on Otsu Image filter."""
    img_arr = sitk.GetArrayFromImage(img)
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_image = otsu_filter.Execute(img)
    otsu_array = sitk.GetArrayFromImage(otsu_image) * (-1) + 1
    slices = robust_bounding_box_3d(otsu_array)
    cropped_img = img_arr[slices]
    # if print_summary:
    #     print(f"Raw: {img_arr.shape}, Cropped: {cropped_img.shape}")
    return sitk.GetImageFromArray(cropped_img)


def crop_background_wrapper(input_dir: os.path, output_dir: os.path):
    assert os.path.isdir(input_dir)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    files = list(
        filter(
            lambda x: x.startswith("NLST") and x.endswith("nii.gz"),
            os.listdir(input_dir),
        )
    )
    for file in tqdm(files):
        print("\n", file)
        img_filepath = os.path.join(input_dir, file)
        img = sitk.ReadImage(img_filepath)
        cropped_img = crop_background(img, print_summary=True)
        sitk.WriteImage(cropped_img, os.path.join(output_dir, "Cropped_" + file))


def affine_registration(
    moving_image: sitk.Image,
    fixed_image: sitk.Image,
    moving_mask: sitk.Image | None = None,
    fixed_mask: sitk.Image | None = None,
) -> (sitk.Image, sitk.Transform):
    min_filter = sitk.MinimumMaximumImageFilter()
    min_filter.Execute(fixed_image)
    min_intensity = min_filter.GetMinimum()

    registration_method = sitk.ImageRegistrationMethod()

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Similarity metric settings.
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.05, seed=1337)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    if moving_mask is not None:
        moving_mask = sitk.Cast(moving_mask, sitk.sitkUInt8)
        registration_method.SetMetricMovingMask(moving_mask)
    if fixed_mask is not None:
        fixed_mask = sitk.Cast(fixed_mask, sitk.sitkUInt8)
        registration_method.SetMetricFixedMask(fixed_mask)

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set the initial moving and optimized transforms.
    optimized_transform = sitk.AffineTransform(fixed_image.GetDimension())
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform)

    # # Need to compose the transformations after registration.
    registration_method.Execute(fixed_image, moving_image)
    final_transform = sitk.CompositeTransform([initial_transform, optimized_transform])

    warped_moving = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        min_intensity,
        moving_image.GetPixelID(),
    )
    return warped_moving, final_transform


if __name__ == "__main__":
    fixed_path = "/home/tsentker/data/rider_lung_ct/RIDER Lung CT/RIDER-1129164940/1.3.6.1.4.1.9328.50.1.216116555221814778114703363464001196508/1.3.6.1.4.1.9328.50.1.48441840081578419409180519840073808100/image.mha"
    moving_path = "/home/tsentker/data/rider_lung_ct/RIDER Lung CT/RIDER-1129164940/1.3.6.1.4.1.9328.50.1.216116555221814778114703363464001196508/1.3.6.1.4.1.9328.50.1.83304264089411327530730818890072724533/image.mha"

    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    start = time.time()
    warped, transform = affine_registration(fixed, moving)
    print(time.time() - start)

    sitk.WriteImage(warped, "/home/tsentker/data/warped_case1_rider.nii.gz")
