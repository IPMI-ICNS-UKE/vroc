from __future__ import annotations

import logging
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from torch.optim import SGD, Adam, NAdam, RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from vroc.blocks import SpatialTransformer
from vroc.helper import compute_tre_numpy, read_landmarks
from vroc.logger import RegistrationLogEntry
from vroc.loss import mse_loss, ncc_loss

logger = logging.getLogger(__name__)


class AffineTransform3d(nn.Module):
    def __init__(self):
        super().__init__()

        self.translation = nn.Parameter(torch.zeros(3))
        self.scale = nn.Parameter(torch.ones(3))
        self.rotation = nn.Parameter(torch.zeros(3))
        self.shear = nn.Parameter(torch.zeros(6))

    @property
    def matrix(self) -> torch.Tensor:
        return AffineTransform3d.create_affine_matrix_3d(
            translation=self.translation,
            scale=self.scale,
            rotation=self.rotation,
            shear=self.shear,
            device=self.translation.device,
        )

    @staticmethod
    def create_affine_matrix_3d(
        translation=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        rotation=(0.0, 0.0, 0.0),
        shear=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # yx, zx, xy, zy, xz, yz
        dtype=torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        # cast to tensors
        translation = torch.as_tensor(translation, dtype=dtype, device=device)
        scale = torch.as_tensor(scale, dtype=dtype, device=device)
        rotation = torch.as_tensor(rotation, dtype=dtype, device=device)
        shear = torch.as_tensor(shear, dtype=dtype, device=device)

        # initialize all matrices with identity
        translation_matrix = torch.diag(torch.ones(4, dtype=dtype, device=device))
        scale_matrix = torch.diag(torch.ones(4, dtype=dtype, device=device))
        rotation_matrix_x = torch.diag(torch.ones(4, dtype=dtype, device=device))
        rotation_matrix_y = torch.diag(torch.ones(4, dtype=dtype, device=device))
        rotation_matrix_z = torch.diag(torch.ones(4, dtype=dtype, device=device))
        shear_matrix = torch.diag(torch.ones(4, dtype=dtype, device=device))

        # translation
        translation_x, translation_y, translation_z = translation
        translation_matrix[0, 3] = translation_x
        translation_matrix[1, 3] = translation_y
        translation_matrix[2, 3] = translation_z

        # scaling
        scale_x, scale_y, scale_z = scale
        scale_matrix[0, 0] = scale_x
        scale_matrix[1, 1] = scale_y
        scale_matrix[2, 2] = scale_z
        # rotation
        # Elemental rotation around one of the axes of the
        # coordinate system (right hand rule).
        theta_x, theta_y, theta_z = rotation

        # shear around x axis
        rotation_matrix_x[1, 1] = torch.cos(theta_x)
        rotation_matrix_x[1, 2] = -torch.sin(theta_x)
        rotation_matrix_x[2, 1] = torch.sin(theta_x)
        rotation_matrix_x[2, 2] = torch.cos(theta_x)
        # shear around y axis
        rotation_matrix_y[0, 0] = torch.cos(theta_y)
        rotation_matrix_y[0, 2] = torch.sin(theta_y)
        rotation_matrix_y[2, 0] = -torch.sin(theta_y)
        rotation_matrix_y[2, 2] = torch.cos(theta_y)
        # shear around z axis
        rotation_matrix_z[0, 0] = torch.cos(theta_z)
        rotation_matrix_z[0, 1] = -torch.sin(theta_z)
        rotation_matrix_z[1, 0] = torch.sin(theta_z)
        rotation_matrix_z[1, 1] = torch.cos(theta_z)

        # matrix multiplication to get the total rotation matrix
        rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x

        # shear
        shear_yx, shear_zx, shear_xy, shear_zy, shear_xz, shear_yz = shear
        # shear along x axis
        shear_matrix[0, 1] = shear_yx
        shear_matrix[0, 2] = shear_zx
        # shear along y axis
        shear_matrix[1, 0] = shear_xy
        shear_matrix[1, 2] = shear_zy
        # shear along z axis
        shear_matrix[2, 0] = shear_xz
        shear_matrix[2, 1] = shear_yz

        transformation_matrix = (
            shear_matrix @ rotation_matrix @ scale_matrix @ translation_matrix
        )

        return transformation_matrix

    def forward(self) -> torch.tensor:
        return self.matrix

    def get_vector_field(self, image: torch.Tensor) -> torch.Tensor:
        device = image.device
        image_spatial_shape = image.shape[2:]

        identity_grid = SpatialTransformer.create_identity_grid(
            shape=image_spatial_shape, device=device
        )

        affine_grid = F.affine_grid(
            self.matrix[None, :3], size=image.shape, align_corners=True
        )

        affine_grid = (affine_grid / 2 + 0.5) * torch.tensor(
            image_spatial_shape[::-1], device=device
        )

        # move channels dim to last position and reverse channels
        affine_grid = affine_grid[..., [2, 1, 0]]
        affine_grid = affine_grid.permute(0, 4, 1, 2, 3)
        # remove identity
        affine_grid = affine_grid - identity_grid

        return affine_grid


def run_affine_registration(
    moving_image: torch.Tensor,
    fixed_image: torch.Tensor,
    moving_mask: torch.Tensor | None = None,
    fixed_mask: torch.Tensor | None = None,
    loss_function: Callable | None = None,
    n_iterations: int = 300,
    step_size: float = 1e-3,
    step_size_reduce_factor: float = 0.5,
    min_step_size: float = 1e-5,
    retry_threshold: float | None = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = moving_image.device
    spatial_image_shape = moving_image.shape[2:]

    initial_step_size = step_size

    if moving_mask is not None and fixed_mask is not None:
        # ROI is union mask
        # roi = moving_mask | fixed_mask
        roi = fixed_mask
        print("Using fixed mask")
    else:
        # ROI is total image
        roi = torch.ones_like(fixed_image, dtype=torch.bool)

    if not loss_function:
        loss_function = mse_loss

    initial_loss = float(loss_function(moving_image, fixed_image, roi))

    # initialize with moving image so that warped image is defined if n_iterations == 0
    warped_image = moving_image
    while True:
        try:
            affine_transform = AffineTransform3d().to(device)
            spatial_transformer = SpatialTransformer(shape=spatial_image_shape).to(
                device
            )
            optimizer = Adam(affine_transform.parameters(), lr=step_size)

            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=step_size_reduce_factor,
                patience=10,
                threshold=0.001,
                min_lr=min_step_size,
            )

            losses = []
            params = {
                "iterations": n_iterations,
                "loss_function": loss_function.__name__,
                "step_size": step_size,
            }
            logger.info(f"Start affine registration with parameters {params}")
            for i in range(n_iterations):
                optimizer.zero_grad()
                affine_matrix = affine_transform.forward()
                warped_image = spatial_transformer.forward(
                    moving_image, affine_matrix[None]
                )

                loss = loss_function(warped_image, fixed_image, roi)

                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                loss = float(loss)
                losses.append(loss)

                log = RegistrationLogEntry(
                    stage="affine",
                    iteration=i,
                    loss=loss,
                    current_step_size=optimizer.param_groups[0]["lr"],
                )
                logger.debug(log)

                relative_change = (loss - initial_loss) / abs(initial_loss)
                if retry_threshold is not None and relative_change > retry_threshold:
                    logger.warning(f"Step size of {step_size} seems to be too large")
                    raise StopIteration
            else:
                # all iterations passed, break while True loop
                break

        except StopIteration:
            # raised if affine registration is aborted due to divergent loss
            # decrease step_size and try again
            step_size = step_size * step_size_reduce_factor
            if step_size < min_step_size:
                # break while True loop
                raise RuntimeError(
                    f"Could not find any step size in range "
                    f"[{initial_step_size}, {min_step_size}] that decreases loss"
                )

            logger.warning(f"Retry with smaller step size of {step_size}")
    logger.info(
        f"Finished affine registration. "
        f"Loss {loss_function.__name__} before/after: "
        f"{initial_loss:.6f}/{losses[-1]:.6f}"
    )

    return warped_image, affine_transform.get_vector_field(moving_image)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import SimpleITK as sitk

    from vroc.logger import LogFormatter

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(LogFormatter())
    logging.basicConfig(handlers=[handler])

    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logging.getLogger("vroc").setLevel(logging.DEBUG)

    def load(
        moving_image_filepath,
        fixed_image_filepath,
        moving_mask_filepath,
        fixed_mask_filepath,
    ):
        moving_image = sitk.ReadImage(moving_image_filepath)
        fixed_image = sitk.ReadImage(fixed_image_filepath)
        moving_mask = sitk.ReadImage(moving_mask_filepath)
        fixed_mask = sitk.ReadImage(fixed_mask_filepath)

        reference_image = fixed_image

        image_spacing = fixed_image.GetSpacing()[::-1]

        moving_image = sitk.GetArrayFromImage(moving_image)
        fixed_image = sitk.GetArrayFromImage(fixed_image)
        moving_mask = sitk.GetArrayFromImage(moving_mask)
        fixed_mask = sitk.GetArrayFromImage(fixed_mask)

        moving_image = np.swapaxes(moving_image, 0, 2)
        fixed_image = np.swapaxes(fixed_image, 0, 2)
        moving_mask = np.swapaxes(moving_mask, 0, 2)
        fixed_mask = np.swapaxes(fixed_mask, 0, 2)

        return (
            moving_image,
            fixed_image,
            moving_mask,
            fixed_mask,
            image_spacing,
            reference_image,
        )

    DEVICE = "cuda:0"
    ROOT_DIR = (
        Path("/home/tsentker/data/learn2reg"),
        Path("/datalake/learn2reg"),
    )
    ROOT_DIR = next(p for p in ROOT_DIR if p.exists())
    FOLDER = "LungCT"

    tres_before = []
    tres_after = []
    for case in range(1, 2):

        fixed_landmarks = read_landmarks(
            f"{ROOT_DIR}/{FOLDER}/keypointsTr/LungCT_{case:04d}_0000.csv",
            sep=",",
        )
        moving_landmarks = read_landmarks(
            f"{ROOT_DIR}/{FOLDER}/keypointsTr/LungCT_{case:04d}_0001.csv",
            sep=",",
        )

        (
            _moving_image,
            _fixed_image,
            _moving_mask,
            _fixed_mask,
            _image_spacing,
            _reference_image,
        ) = load(
            moving_image_filepath=f"{ROOT_DIR}/{FOLDER}/imagesTr/LungCT_{case:04d}_0001.nii.gz",
            fixed_image_filepath=f"{ROOT_DIR}/{FOLDER}/imagesTr/LungCT_{case:04d}_0000.nii.gz",
            moving_mask_filepath=f"{ROOT_DIR}/{FOLDER}/masksTr/LungCT_{case:04d}_0001.nii.gz",
            fixed_mask_filepath=f"{ROOT_DIR}/{FOLDER}/masksTr/LungCT_{case:04d}_0000.nii.gz",
        )

        # _moving_mask = binary_dilation(
        #     _moving_mask.astype(np.uint8), iterations=1
        # ).astype(bool)
        # _fixed_mask = binary_dilation(
        #     _fixed_mask.astype(np.uint8), iterations=1
        # ).astype(bool)

        moving_image = torch.as_tensor(_moving_image[None, None], device=DEVICE)
        fixed_image = torch.as_tensor(_fixed_image[None, None], device=DEVICE)
        moving_mask = torch.as_tensor(
            _moving_mask[None, None], dtype=torch.bool, device=DEVICE
        )
        fixed_mask = torch.as_tensor(
            _fixed_mask[None, None], dtype=torch.bool, device=DEVICE
        )

        warped_image, vector_field = run_affine_registration(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            loss_function=mse_loss,
            step_size=1e-3,
        )

        _vector_field = vector_field.detach().cpu().numpy().squeeze()
        _warped_image = warped_image.detach().cpu().numpy().squeeze()

        tre_before = compute_tre_numpy(
            moving_landmarks=moving_landmarks,
            fixed_landmarks=fixed_landmarks,
            vector_field=None,
            image_spacing=(1.5,) * 3,
        )
        tre_after = compute_tre_numpy(
            moving_landmarks=moving_landmarks,
            fixed_landmarks=fixed_landmarks,
            vector_field=_vector_field,
            image_spacing=(1.5,) * 3,
        )

        print(
            f"NLST_0{case}: "
            f"tre_before={np.mean(tre_before):.2f}, "
            f"tre_after={np.mean(tre_after):.2f}"
        )

        tres_before.append(np.mean(tre_before))
        tres_after.append(np.mean(tre_after))

        mid_slice = _fixed_image.shape[1] // 2
        clim = (-800, 200)
        fig, ax = plt.subplots(1, 5, sharex=True, sharey=True)
        ax[0].imshow(_fixed_image[:, mid_slice, :], clim=clim)
        ax[1].imshow(_moving_image[:, mid_slice, :], clim=clim)
        ax[2].imshow(_warped_image[:, mid_slice, :], clim=clim)
        ax[3].imshow(
            _moving_image[:, mid_slice, :] - _fixed_image[:, mid_slice, :],
            clim=(-500, 500),
            cmap="seismic",
        )
        ax[4].imshow(
            _warped_image[:, mid_slice, :] - _fixed_image[:, mid_slice, :],
            clim=(-500, 500),
            cmap="seismic",
        )

        fig.suptitle(f"NLST_{case:04d}")

    print(f"before: mean TRE={np.mean(tres_before)}, std TRE={np.std(tres_before)}")
    print(f"after: mean TRE={np.mean(tres_after)}, std TRE={np.std(tres_after)}")
