import logging
from pathlib import Path
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch

from vroc.blocks import SpatialTransformer
from vroc.convert import as_tensor
from vroc.interpolation import resize_spacing
from vroc.logger import init_fancy_logging
from vroc.loss import mse_loss, ncc_loss
from vroc.preprocessing import crop_or_pad
from vroc.registration import VrocRegistration

if __name__ == "__main__":
    init_fancy_logging()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging.getLogger("vroc").setLevel(logging.INFO)
    logging.getLogger("vroc.models").setLevel(logging.INFO)

    device = "cuda:0"

    case_ids = [
        7,
        9,
        10,
        13,
        16,
        17,
        43,
        44,
        45,
        47,
        48,
        49,
        54,
        56,
        57,
        58,
        64,
        65,
        66,
        67,
        74,
        92,
        99,
        114,
    ]
    for case_id in case_ids:
        case = f"SP_{case_id:03d}"
        path = Path(f"/home/tsentker/data/NWU/converted/{case}")

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
        nect = sitk.Mask(nect, nect_brain_mask)

        dwi_arr = sitk.GetArrayFromImage(dwi)
        nect_arr = sitk.GetArrayFromImage(nect)
        nect_brain_mask_arr = sitk.GetArrayFromImage(nect_brain_mask)
        dwi_lesion_mask_arr = sitk.GetArrayFromImage(dwi_lesion_mask)
        dwi_spacing = (dwi.GetSpacing()[2], dwi.GetSpacing()[0], dwi.GetSpacing()[1])
        nect_spacing = (
            nect.GetSpacing()[2],
            nect.GetSpacing()[0],
            nect.GetSpacing()[1],
        )

        out_spacing = (3.0, 1.0, 1.0)

        dwi_arr = resize_spacing(
            image=dwi_arr,
            input_image_spacing=dwi_spacing,
            output_image_spacing=out_spacing,
        )
        initial_shape = dwi_arr.shape
        dwi_lesion_mask_arr = resize_spacing(
            image=dwi_lesion_mask_arr,
            input_image_spacing=dwi_spacing,
            output_image_spacing=out_spacing,
            order=0,
        )

        nect_arr = resize_spacing(
            image=nect_arr,
            input_image_spacing=nect_spacing,
            output_image_spacing=out_spacing,
        )
        nect_brain_mask_arr = resize_spacing(
            image=nect_brain_mask_arr,
            input_image_spacing=nect_spacing,
            output_image_spacing=out_spacing,
            order=0,
        )

        if nect_arr.shape != dwi_arr.shape:
            out_shape = [
                int(np.max((i, j))) for i, j in zip(nect_arr.shape, dwi_arr.shape)
            ]
            nect_arr = crop_or_pad(
                image=nect_arr,
                mask=None,
                target_shape=tuple(out_shape),
                image_pad_value=0,
            )[0]
            dwi_arr = crop_or_pad(
                image=dwi_arr,
                mask=None,
                target_shape=tuple(out_shape),
                image_pad_value=0,
            )[0]
            nect_brain_mask_arr = crop_or_pad(
                image=nect_brain_mask_arr,
                mask=None,
                target_shape=tuple(out_shape),
                image_pad_value=0,
            )[0]
            dwi_lesion_mask_arr = crop_or_pad(
                image=dwi_lesion_mask_arr,
                mask=None,
                target_shape=tuple(out_shape),
                image_pad_value=0,
            )[0]

        device = torch.device(device)
        n_dims = 3

        dwi_image = as_tensor(
            dwi_arr, n_dim=n_dims + 2, dtype=torch.float16, device=device
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
            "tau": 0,
            "n_levels": 2,
            "largest_scale_factor": 1,
        }

        reg_result = registration.register(
            moving_image=nect_image,
            fixed_image=dwi_image,
            # fixed_mask=nect_brain_mask,
            image_spacing=out_spacing,
            register_affine=True,
            affine_loss_function=ncc_loss,
            affine_iterations=300,
            affine_step_size=0.01,
            force_type="ncc",
            gradient_type="passive",
            valid_value_range=(-1024, 3071),
            early_stopping_delta=0.00,
            early_stopping_window=100,
            default_parameters=params,
        )

        dim_0 = initial_shape[0] // 2
        dim_1 = initial_shape[1] // 2
        dim_2 = initial_shape[2] // 2

        fig, ax = plt.subplots(nrows=3, ncols=3, sharex="col", sharey="row")
        fig.suptitle(f"Case {case_id}", fontsize=16)
        ax[0, 0].imshow(
            reg_result.moving_image[:, dim_1, :],
            aspect="auto",
            vmin=0,
            vmax=80,
            cmap="gray",
        )
        ax[0, 0].set_title("moving (nect)", fontsize=10)
        ax[0, 1].imshow(
            reg_result.warped_moving_image[:, dim_1, :],
            aspect="auto",
            vmin=0,
            vmax=80,
            cmap="gray",
        )
        ax[0, 1].set_title("warped (nect)", fontsize=10)
        ax[0, 2].imshow(dwi_arr[:, dim_1, :], aspect="auto", cmap="gray")
        ax[0, 2].set_title("fixed (dwi)", fontsize=10)
        ax[1, 0].imshow(
            reg_result.moving_image[dim_0, :, :],
            aspect="auto",
            vmin=0,
            vmax=80,
            cmap="gray",
        )
        ax[1, 1].imshow(
            reg_result.warped_moving_image[dim_0, :, :],
            aspect="auto",
            vmin=0,
            vmax=80,
            cmap="gray",
        )
        ax[1, 2].imshow(dwi_arr[dim_0, :, :], aspect="auto", cmap="gray")
        ax[2, 0].imshow(
            reg_result.moving_image[:, :, dim_2],
            aspect="auto",
            vmin=0,
            vmax=80,
            cmap="gray",
        )
        ax[2, 1].imshow(
            reg_result.warped_moving_image[:, :, dim_2],
            aspect="auto",
            vmin=0,
            vmax=80,
            cmap="gray",
        )
        ax[2, 2].imshow(dwi_arr[:, :, dim_2], aspect="auto", cmap="gray")
        fig.show()

        warped = reg_result.warped_moving_image
        warped = crop_or_pad(
            image=warped,
            mask=None,
            target_shape=initial_shape,
            image_pad_value=0,
        )[0]

        out_warped = sitk.GetImageFromArray(warped)
        out_warped.SetSpacing((1, 1, 3))

        dwi_arr = crop_or_pad(
            image=dwi_arr,
            mask=None,
            target_shape=initial_shape,
            image_pad_value=0,
        )[0]
        out_dwi = sitk.GetImageFromArray(dwi_arr)
        out_dwi.SetSpacing((1, 1, 3))

        dwi_lesion_mask_arr = crop_or_pad(
            image=dwi_lesion_mask_arr,
            mask=None,
            target_shape=initial_shape,
            image_pad_value=0,
        )[0]
        out_dwi_lesion_mask = sitk.GetImageFromArray(dwi_lesion_mask_arr)
        out_dwi_lesion_mask.SetSpacing((1, 1, 3))

        Path(f"/home/tsentker/data/NWU/out/{case}").mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(
            out_warped, f"/home/tsentker/data/NWU/out/{case}/nect_warped.nii.gz"
        )
        sitk.WriteImage(out_dwi, f"/home/tsentker/data/NWU/out/{case}/dwi.nii.gz")
        sitk.WriteImage(
            out_dwi_lesion_mask,
            f"/home/tsentker/data/NWU/out/{case}/dwi_lesion_mask.nii.gz",
        )
