from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vroc.blocks import SpatialTransformer, DemonForces3d, GaussianSmoothing3d


class TrainableVarRegBlock(nn.Module):
    def __init__(
        self,
        patch_shape,
        iterations: Tuple[int, ...] = (100, 100),
        tau: float = 1.0,
        regularization_sigma=(1.0, 1.0, 1.0),
        disable_correction: bool = True,
        scale_factors: Tuple[float, ...] = (0.5, 1.0),
        early_stopping_delta: float = 0.0,
        early_stopping_window: int = 10,
    ):
        super().__init__()
        self.patch_shape = patch_shape
        self.iterations = iterations
        self.tau = tau
        self.regularization_sigma = regularization_sigma
        self.disable_correction = disable_correction
        self.scale_factors = scale_factors

        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_window = early_stopping_window
        self._metrics = []

        self._full_size_spatial_transformer = SpatialTransformer(shape=patch_shape)

        for i_level, scale_factor in enumerate(scale_factors):
            scaled_patch_shape = tuple(int(s * scale_factor) for s in self.patch_shape)

            self.add_module(
                name=f"spatial_transformer_level_{i_level}",
                module=SpatialTransformer(shape=scaled_patch_shape),
            )

        self._demon_forces_layer = DemonForces3d()

        self._regularization_layer = GaussianSmoothing3d(
            sigma=self.regularization_sigma, sigma_cutoff=2
        )

        self._vector_field_updater = nn.Sequential(
            nn.Conv3d(
                # 3 DVF channels, 2x1 image channels
                in_channels=5,
                out_channels=32,
                kernel_size=(3, 3, 3),
                padding="same",
                bias=False,
            ),
            nn.InstanceNorm3d(num_features=32),
            nn.Mish(),
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3, 3),
                padding="same",
                bias=False,
            ),
            nn.InstanceNorm3d(num_features=64),
            nn.Mish(),
            nn.Conv3d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3, 3, 3),
                padding="same",
                bias=False,
            ),
            nn.InstanceNorm3d(num_features=32),
            nn.Mish(),
            nn.Conv3d(
                in_channels=32,
                out_channels=3,
                kernel_size=(3, 3, 3),
                padding="same",
                bias=True,
            ),
        )

    @property
    def config(self):
        return dict(
            iterations=self.iterations,
            tau=self.tau,
            regularization_sigma=self.regularization_sigma,
        )

    def _check_early_stopping(self, metrics: List[float]):
        if (
            self.early_stopping_delta == 0
            or len(metrics) < self.early_stopping_window + 1
        ):
            return False

        rel_change = (metrics[-self.early_stopping_window] - metrics[-1]) / (
            metrics[-self.early_stopping_window] + 1e-9
        )
        # print(f'Rel. change: {rel_change}')

        if rel_change < self.early_stopping_delta:
            return True
        return False

    def _check_early_stopping_average_improvement(self, metrics: List[float]):
        if (
            self.early_stopping_delta == 0
            or len(metrics) < self.early_stopping_window + 1
        ):
            return False

        window = np.array(metrics[-self.early_stopping_window :])
        window_rel_changes = 1 - window[1:] / window[:-1]

        if window_rel_changes.mean() < self.early_stopping_delta:
            # print(f'Rel. change: {window_rel_changes}')
            return True
        return False

    def _perform_scaling(self, image, mask, moving, scale_factor: float = 1.0):
        image = F.interpolate(
            image, scale_factor=scale_factor, mode="trilinear", align_corners=False
        )
        mask = F.interpolate(mask, scale_factor=scale_factor, mode="nearest")
        moving = F.interpolate(
            moving,
            scale_factor=scale_factor,
            mode="trilinear",
            align_corners=False,
        )

        return image, mask, moving

    def _match_vector_field(self, vector_field, image):
        if vector_field.shape[2:] == image.shape[2:]:
            return vector_field

        vector_field_shape = vector_field.shape

        vector_field = F.interpolate(
            vector_field, size=image.shape[2:], mode="trilinear", align_corners=False
        )

        scale_factor = torch.ones((1, 3, 1, 1, 1), device=vector_field.device)
        scale_factor[0, :, 0, 0, 0] = torch.tensor(
            [s1 / s2 for (s1, s2) in zip(image.shape[2:], vector_field_shape[2:])]
        )

        return vector_field * scale_factor

    def warp_moving(self, image, mask, moving):
        # register moving image onto fixed image
        full_size_moving = moving

        vector_field = None
        with torch.no_grad():

            for i_level, (scale_factor, iterations) in enumerate(
                zip(self.scale_factors, self.iterations)
            ):
                metrics = []

                (scaled_image, scaled_mask, scaled_moving) = self._perform_scaling(
                    image, mask, moving, scale_factor=scale_factor
                )

                if vector_field is None:
                    vector_field = torch.zeros(
                        scaled_image.shape[:1] + (3,) + scaled_image.shape[2:],
                        device=moving.device,
                    )
                elif vector_field.shape[2:] != scaled_image.shape[2:]:
                    vector_field = self._match_vector_field(vector_field, scaled_image)

                spatial_transformer = self.get_submodule(
                    f"spatial_transformer_level_{i_level}",
                )
                for i in range(iterations):
                    # print(f'*** ITERATION {i + 1} ***')

                    warped_moving = spatial_transformer(
                        scaled_moving, vector_field
                    )

                    metrics.append(float(F.mse_loss(scaled_image, warped_moving)))

                    if self._check_early_stopping_average_improvement(metrics):
                        # print(f'Early stopping at iter {i}')
                        break

                    forces = self._demon_forces_layer(warped_moving, scaled_image)
                    # mask forces with artifact mask (artifact = 0, valid = 1)

                    vector_field -= self.tau * (forces * scaled_mask)
                    vector_field = self._regularization_layer(vector_field)

                # print(f'ITERATION {i + 1}: Metric: {metrics[-1]}')
        if self.disable_correction:
            corrected_vector_field = vector_field
        else:
            # update/correct DVF at artifact region
            masked_image = image * mask
            stacked = torch.cat((vector_field, masked_image, warped_moving), dim=1)

            vector_field_correction = self._vector_field_updater(stacked)
            corrected_vector_field = vector_field + vector_field_correction

            if not corrected_vector_field.isfinite().all():
                raise ValueError()

            corrected_vector_field = self._regularization_layer(corrected_vector_field)

        vector_field = self._match_vector_field(vector_field, full_size_moving)
        corrected_vector_field = self._match_vector_field(
            corrected_vector_field, full_size_moving
        )

        corrected_warped_moving = self._full_size_spatial_transformer(
            full_size_moving, corrected_vector_field
        )
        warped_moving = self._full_size_spatial_transformer(
            full_size_moving, vector_field
        )

        return (
            corrected_warped_moving,
            warped_moving,
            corrected_vector_field,
            vector_field,
        )

    def forward(self, image, mask, moving):
        (
            corrected_warped_moving,
            warped_moving,
            corrected_vector_field,
            vector_field,
        ) = self.warp_moving(image, mask, moving)

        return (
            corrected_warped_moving,
            warped_moving,
            corrected_vector_field,
            vector_field,
        )


if __name__ == "__main__":

    import SimpleITK as sitk
    import os
    import torch
    from vroc.helper import read_landmarks, transform_landmarks, target_registration_errors


    def load_and_preprocess(filepath):
        filepath = str(filepath)
        image = sitk.ReadImage(filepath, sitk.sitkFloat32)
        image = sitk.GetArrayFromImage(image)
        # image = image.swapaxes(0, 2)

        return image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = '/home/tsentker/data/dirlab2022/data/Case06Pack'
    out_path = '/home/tsentker/Documents/results/varreg-on-crack/'

    # fixed = np.zeros((40, 40, 40))
    # moving = np.zeros((40, 40, 40))
    # fixed[18:22, 18:22, 20] = 1.0
    # moving[22:24, 22:24, 20] = 1.0
    #
    # patch_shape = fixed.shape
    # fixed_ = torch.from_numpy(fixed.copy())
    # fixed_ = fixed_[None, None, :].float().to(device)
    # moving_ = torch.from_numpy(moving.copy())
    # moving_ = moving_[None, None, :].float().to(device)
    # mask = torch.ones_like(fixed_).float().to(device)
    # model = TrainableVarRegBlock(patch_shape=patch_shape).to(device)
    # out = model.forward(fixed_, mask, moving_)
    #
    # warped_image = out[1].cpu().detach().numpy()
    # warped_image = np.squeeze(warped_image)
    # warped_image = sitk.GetImageFromArray(warped_image)
    # sitk.WriteImage(warped_image, os.path.join(out_path, 'warped.mha'))
    #
    # vf = out[3].cpu().detach().numpy()
    # vf = np.squeeze(vf)
    # vf = vf.swapaxes(0, 3)
    # vf = vf.swapaxes(1, 2)
    # vf = sitk.GetImageFromArray(vf, isVector=True)
    # vf = sitk.Cast(vf, sitk.sitkVectorFloat64)
    # inverter = sitk.IterativeInverseDisplacementFieldImageFilter()
    # vf = inverter.Execute(vf)
    # warper = sitk.WarpImageFilter()
    # sitk_warped = warper.Execute(sitk.GetImageFromArray(moving), vf)
    # sitk.WriteImage(sitk_warped, os.path.join(out_path, 'sitk_warped.mha'))
    # sitk.WriteImage(vf, os.path.join(out_path, 'vf.mha'))

    fixed = load_and_preprocess(os.path.join(data_path, 'Images', 'phase_0.mha'))
    moving = load_and_preprocess(os.path.join(data_path, 'Images', 'phase_5.mha'))

    fixed = torch.from_numpy(fixed.copy())
    patch_shape = fixed.shape
    fixed = fixed[None, None, :].float().to(device)
    moving = torch.from_numpy(moving.copy())
    moving = moving[None, None, :].float().to(device)
    mask = torch.ones_like(fixed).float().to(device)
    model = TrainableVarRegBlock(patch_shape=patch_shape).to(device)
    out = model.forward(fixed, mask, moving)

    orig_ref = sitk.ReadImage(os.path.join(data_path, 'Images', 'phase_0.mha'))
    orig_moving = sitk.ReadImage(os.path.join(data_path, 'Images', 'phase_5.mha'))

    warped_image = out[1].cpu().detach().numpy()
    warped_image = np.squeeze(warped_image)
    # warped_image = warped_image.swapaxes(0, 2)
    warped_image = sitk.GetImageFromArray(warped_image)
    # warped_image.CopyInformation(orig_ref)
    sitk.WriteImage(warped_image, os.path.join(out_path, 'warped.mha'))

    vf = out[3].cpu().detach().numpy()
    vf = np.squeeze(vf)
    vf = np.rollaxis(vf, 0, vf.ndim)
    # vf = vf.swapaxes(1, 2)
    # for i, sp in enumerate(orig_ref.GetSpacing()):
    #     vf[:, :, :, i] = vf[:, :, :, i] * sp
    vf = sitk.GetImageFromArray(vf, isVector=True)
    inverter = sitk.IterativeInverseDisplacementFieldImageFilter()
    vf = inverter.Execute(vf)
    # vf.CopyInformation(orig_ref)
    vf = sitk.Cast(vf, sitk.sitkVectorFloat64)
    warper = sitk.WarpImageFilter()
    # warper.SetOutputParameteresFromImage(orig_moving)
    orig_moving.CopyInformation(vf)
    sitk_warped = warper.Execute(orig_moving, vf)
    sitk.WriteImage(sitk_warped, os.path.join(out_path, 'sitk_warped.mha'))
    sitk.WriteImage(vf, os.path.join(out_path, 'vf.mha'))

    fixed_LM = read_landmarks(os.path.join(data_path, 'extremePhases', 'case6_dirLab300_T00_xyz.txt'))
    fixed_LM = transform_landmarks(fixed_LM, orig_ref)
    moving_LM = read_landmarks(os.path.join(data_path, 'extremePhases', 'case6_dirLab300_T50_xyz.txt'))
    moving_LM = transform_landmarks(moving_LM, orig_ref)

    vf_transformed = sitk.DisplacementFieldTransform(vf)
    TRE = np.mean(target_registration_errors(vf_transformed, moving_LM, fixed_LM))


