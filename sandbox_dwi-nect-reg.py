import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch

from vroc.blocks import SpatialTransformer
from vroc.convert import as_tensor
from vroc.interpolation import resize_spacing
from vroc.logger import init_fancy_logging
from vroc.loss import ncc_loss
from vroc.registration import VrocRegistration

if __name__ == "__main__":
    init_fancy_logging()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging.getLogger("vroc").setLevel(logging.INFO)
    logging.getLogger("vroc.models").setLevel(logging.INFO)

    device = "cuda:0"

    case_id = 7
    case = f"SP_{case_id:03d}"
    path = Path(f"/home/tsentker/data/NWU/converted/{case}")

    # ct.create_ct_segmentations(image_filepath=str(path / "dwi.nii.gz"),
    #                            output_folder=str(path / "total_seg_dwi"),
    #                            models=['total'])

    dwi = sitk.ReadImage(str(path / "dwi.nii.gz"), outputPixelType=sitk.sitkFloat32)
    nect = sitk.ReadImage(
        str(path / "nect.nii.gz"),
    )
    nect_brain_mask = sitk.ReadImage(
        str(path / "total_seg" / "brain.nii.gz"),
    )
    dwi_lesion_mask = sitk.ReadImage(
        str(path / "dwi_lesion_mask.nii.gz"),
    )
    nect = nect[:, :, :, 0]
    nect = sitk.Mask(nect, nect_brain_mask)

    dwi_arr = sitk.GetArrayFromImage(dwi)
    nect_arr = sitk.GetArrayFromImage(nect)
    nect_brain_mask_arr = sitk.GetArrayFromImage(nect_brain_mask)
    dwi_lesion_mask_arr = sitk.GetArrayFromImage(dwi_lesion_mask)
    dwi_spacing = (dwi.GetSpacing()[2], dwi.GetSpacing()[0], dwi.GetSpacing()[1])
    nect_spacing = (nect.GetSpacing()[2], nect.GetSpacing()[0], nect.GetSpacing()[1])

    dwi_arr = resize_spacing(
        image=dwi_arr,
        input_image_spacing=dwi_spacing,
        output_image_spacing=(1, 1, 1),
    )
    dwi_lesion_mask_arr = resize_spacing(
        image=dwi_lesion_mask_arr,
        input_image_spacing=dwi_spacing,
        output_image_spacing=(1, 1, 1),
        order=0,
    )
    nect_arr = resize_spacing(
        image=nect_arr,
        input_image_spacing=nect_spacing,
        output_image_spacing=(1, 1, 1),
    )
    nect_brain_mask_arr = resize_spacing(
        image=nect_brain_mask_arr,
        input_image_spacing=nect_spacing,
        output_image_spacing=(1, 1, 1),
    )
    shape_diff = [i - j for i, j in zip(dwi_arr.shape, nect_arr.shape)]

    tmp_array = np.zeros_like(dwi_arr)
    tmp_array_mask = np.zeros_like(dwi_arr)
    # TODO: PLEASE fix +1/-1 randomness
    tmp_array[
        int(shape_diff[0] / 2) : dwi_arr.shape[0] - int(shape_diff[0] / 2 + 1),
        int(shape_diff[1] / 2 + 1) : int(-shape_diff[1] / 2),
        int(shape_diff[2] / 2 + 1) : int(-shape_diff[2] / 2),
    ] = nect_arr
    nect_arr = tmp_array
    tmp_array_mask[
        int(shape_diff[0] / 2) : dwi_arr.shape[0] - int(shape_diff[0] / 2 + 1),
        int(shape_diff[1] / 2 + 1) : int(-shape_diff[1] / 2),
        int(shape_diff[2] / 2 + 1) : int(-shape_diff[2] / 2),
    ] = nect_brain_mask_arr
    nect_brain_mask_arr = tmp_array_mask

    device = torch.device(device)
    n_dims = 3

    dwi_image = as_tensor(dwi_arr, n_dim=n_dims + 2, dtype=torch.float16, device=device)
    dwi_lesion_mask = as_tensor(
        dwi_lesion_mask_arr, n_dim=n_dims + 2, dtype=torch.bool, device=device
    )
    nect_image = as_tensor(
        nect_arr, n_dim=n_dims + 2, dtype=torch.float16, device=device
    )
    nect_brain_mask = as_tensor(
        nect_brain_mask_arr, n_dim=n_dims + 2, dtype=torch.float16, device=device
    )

    registration = VrocRegistration(
        roi_segmenter=None,
        feature_extractor=None,
        parameter_guesser=None,
        device="cuda:0",
    )

    params = {
        "iterations": 0,
        "tau": 25,
        "n_levels": 2,
        "largest_scale_factor": 1,
    }

    reg_result = registration.register(
        moving_image=nect_image,
        fixed_image=dwi_image,
        fixed_mask=nect_brain_mask,
        register_affine=True,
        affine_loss_function=ncc_loss,
        force_type="ncc",
        gradient_type="passive",
        valid_value_range=(-1024, 3071),
        early_stopping_delta=0.00,
        early_stopping_window=100,
        default_parameters=params,
    )

    # spatial_image_shape = nect_image.shape[2:]
    # spatial_transformer = SpatialTransformer(shape=spatial_image_shape).to(
    #     device
    # )
    # warped_lesion_mask = spatial_transformer.forward(
    #     dwi_lesion_mask, as_tensor(reg_result.composed_vector_field, n_dim=n_dims + 1, dtype=torch.float32, device=device)
    # )
    # warped_lesion_mask = warped_lesion_mask.cpu().numpy().squeeze().squeeze()

    fig, ax = plt.subplots(3, 3)
    ax[0, 0].imshow(reg_result.fixed_image[:, 120, :], aspect="auto")
    ax[0, 1].imshow(reg_result.warped_moving_image[:, 120, :], aspect="auto")
    ax[0, 2].imshow(dwi_arr[:, 120, :], aspect="auto")
    ax[1, 0].imshow(reg_result.fixed_image[80, :, :], aspect="auto")
    ax[1, 1].imshow(reg_result.warped_moving_image[80, :, :], aspect="auto")
    ax[1, 2].imshow(dwi_arr[80, :, :], aspect="auto")
    ax[2, 0].imshow(reg_result.fixed_image[:, :, 120], aspect="auto")
    ax[2, 1].imshow(reg_result.warped_moving_image[:, :, 120], aspect="auto")
    ax[2, 2].imshow(dwi_arr[:, :, 120], aspect="auto")
    fig.show()

    # warped_lesion_mask = resize_spacing(
    #     warped_lesion_mask, input_image_spacing=(1, 1, 1), output_image_spacing=nect_spacing
    # )

    warped = reg_result.warped_moving_image
    warped_no_resample = reg_result.warped_moving_image
    warped = resize_spacing(
        warped, input_image_spacing=(1, 1, 1), output_image_spacing=dwi_spacing
    )
    out = sitk.GetImageFromArray(warped)
    out.CopyInformation(dwi)
    # out_mask = sitk.GetImageFromArray(warped_lesion_mask.astype(int))
    # out_mask.CopyInformation(dwi)

    sitk.WriteImage(out, f"/home/tsentker/data/NWU/out/{case}/nect_warped.nii.gz")
    warped_no_resample = sitk.GetImageFromArray(warped_no_resample)
    dwi_lesion_mask_no_res = sitk.GetImageFromArray(dwi_lesion_mask_arr)
    dwi_no_res = sitk.GetImageFromArray(dwi_arr)

    sitk.WriteImage(
        dwi_lesion_mask_no_res,
        f"/home/tsentker/data/NWU/out/{case}/dwi_lesion_mask_no_resample.nii.gz",
    )
    sitk.WriteImage(
        dwi_no_res, f"/home/tsentker/data/NWU/out/{case}/dwi_no_resample.nii.gz"
    )
    sitk.WriteImage(
        warped_no_resample,
        f"/home/tsentker/data/NWU/out/{case}/nect_warped_no_resample.nii.gz",
    )

    # sitk.WriteImage(out_mask, f"/home/tsentker/data/NWU/out/{case}/dwi_lesion_mask_warped.nii.gz")
