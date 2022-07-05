import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


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


def affine_registration(fixed: sitk.Image, moving: sitk.Image) -> sitk.Image:
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg_method = sitk.ImageRegistrationMethod()
    reg_method.SetMetricAsMeanSquares()
    reg_method.SetMetricSamplingStrategy(reg_method.RANDOM)
    reg_method.SetMetricSamplingPercentage(0.01)
    reg_method.SetInterpolator(sitk.sitkLinear)

    # reg_method.SetOptimizerAsGradientDescent(learning_rate=1.0, numberOfIterations=100)
    reg_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 1])
    reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 0])
    reg_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = reg_method.Execute(fixed, moving)
    warped_moving = sitk.Resample(
        moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    return warped_moving


def multires_registration(
    fixed: sitk.Image, moving: sitk.Image, mask_fixed, mask_moving
) -> sitk.Image:
    min_filter = sitk.MinimumMaximumImageFilter()
    min_filter.Execute(fixed)
    min_intensity = min_filter.GetMinimum()

    mask_filter = sitk.MaskImageFilter()
    masked_fixed = mask_filter.Execute(fixed, mask_fixed)
    masked_moving = mask_filter.Execute(moving, mask_moving)

    initial_transform = sitk.CenteredTransformInitializer(
        masked_fixed,
        masked_moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetInterpolator(sitk.sitkBSpline)
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.5,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    optimized_transform = sitk.AffineTransform(fixed.GetDimension())
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform, inPlace=False)

    optimized_transform = registration_method.Execute(masked_fixed, masked_moving)
    warped_moving = sitk.Resample(
        moving,
        fixed,
        optimized_transform,
        sitk.sitkLinear,
        min_intensity,
        moving.GetPixelID(),
    )

    warped_moving.CopyInformation(fixed)
    return warped_moving


if __name__ == "__main__":

    fixed_path = (
        "/home/tsentker/data/learn2reg/NLST_fixed/imagesTr/NLST_0019_0000.nii.gz"
    )
    mask_fixed_path = (
        "/home/tsentker/data/learn2reg/NLST_fixed/masksTr/NLST_0019_0000.nii.gz"
    )
    mask_moving_path = (
        "/home/tsentker/data/learn2reg/NLST_fixed/masksTr/NLST_0019_0000.nii.gz"
    )
    moving_path = (
        "/home/tsentker/data/learn2reg/NLST_fixed/imagesTr/NLST_0019_0001.nii.gz"
    )

    fixed = sitk.ReadImage(fixed_path)
    moving = sitk.ReadImage(moving_path)
    mask_fixed = sitk.ReadImage(mask_fixed_path, sitk.sitkUInt8)
    mask_moving = sitk.ReadImage(mask_moving_path, sitk.sitkUInt8)

    start = time.time()
    warped = multires_registration(fixed, moving, mask_fixed, mask_moving)
    print(time.time() - start)

    mse_pre = (
        np.square(
            sitk.GetArrayFromImage(fixed) * sitk.GetArrayFromImage(mask_fixed)
            - sitk.GetArrayFromImage(moving) * sitk.GetArrayFromImage(mask_fixed)
        )
    ).mean(axis=None)
    mse_post = (
        np.square(
            sitk.GetArrayFromImage(fixed) * sitk.GetArrayFromImage(mask_fixed)
            - sitk.GetArrayFromImage(warped) * sitk.GetArrayFromImage(mask_fixed)
        )
    ).mean(axis=None)

    sitk.WriteImage(
        warped, "/home/tsentker/data/learn2reg/NLST_fixed/29_to_01_affine.nii.gz"
    )
