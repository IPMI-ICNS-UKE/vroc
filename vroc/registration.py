from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Literal, Sequence, Tuple

import numpy as np
import torch

from vroc.affine import run_affine_registration
from vroc.common_types import (
    ArrayOrTensor,
    FloatTuple2D,
    FloatTuple3D,
    MaybeSequence,
    Number,
    PathLike,
    TorchDevice,
)
from vroc.convert import as_tensor
from vroc.decorators import convert, timing
from vroc.logger import LoggerMixin
from vroc.models import VariationalRegistration


@dataclass
class RegistrationResult:
    # initial images
    moving_image: np.ndarray | torch.Tensor
    fixed_image: np.ndarray | torch.Tensor

    warped_moving_image: np.ndarray | torch.Tensor
    composed_vector_field: np.ndarray | torch.Tensor
    vector_fields: List[np.ndarray | torch.Tensor]

    # masks
    moving_mask: np.ndarray | torch.Tensor | None = None
    warped_moving_mask: np.ndarray | torch.Tensor = None
    fixed_mask: np.ndarray | torch.Tensor | None = None

    warped_affine_moving_image: np.ndarray | torch.Tensor | None = None
    warped_affine_moving_mask: np.ndarray | torch.Tensor | None = None

    # keypoints
    moving_keypoints: np.ndarray | torch.Tensor | None = None
    fixed_keypoints: np.ndarray | torch.Tensor | None = None

    debug_info: dict | None = None
    level: int | None = None
    scale_factor: float | None = None
    iteration: int | None = None
    is_final_result: bool = True

    def _get_castable_variables(self) -> List[str]:
        valid_instances = (torch.Tensor, np.ndarray)
        castable = []
        for name, value in self.__dir__():
            if isinstance(value, valid_instances):
                castable.append(name)

        return castable

    def _cast(self, cast_function: Callable):
        self.moving_image = cast_function(self.moving_image)
        self.fixed_image = cast_function(self.fixed_image)

        self.warped_moving_image = cast_function(self.warped_moving_image)
        self.composed_vector_field = cast_function(self.composed_vector_field)
        self.vector_fields = cast_function(self.vector_fields)

        self.moving_mask = cast_function(self.moving_mask)
        self.warped_moving_mask = cast_function(self.warped_moving_mask)
        self.fixed_mask = cast_function(self.fixed_mask)

        self.warped_affine_moving_image = cast_function(self.warped_affine_moving_image)
        self.warped_affine_moving_mask = cast_function(self.warped_affine_moving_mask)

        self.moving_keypoints = cast_function(self.moving_keypoints)
        self.fixed_keypoints = cast_function(self.fixed_keypoints)

    def to_numpy(self):
        def cast_function(
            tensor: torch.Tensor | Sequence[torch.Tensor],
        ) -> np.ndarray | List[np.ndarray] | None:
            def _cast_tensor(tensor: torch.Tensor) -> np.ndarray:
                return tensor.detach().cpu().numpy().squeeze()

            if tensor is None or isinstance(tensor, np.ndarray):
                return tensor
            elif isinstance(tensor, (tuple, list)):
                return [_cast_tensor(t) for t in tensor]
            else:
                return _cast_tensor(tensor)

        self._cast(cast_function)

    def to(self, device: TorchDevice):
        def cast_function(
            tensor: torch.Tensor | Sequence[torch.Tensor],
        ) -> torch.Tensor | List[torch.Tensor] | None:
            def _cast_tensor(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.to(device)

            if isinstance(tensor, (tuple, list)):
                return [_cast_tensor(t) for t in tensor]
            else:
                return _cast_tensor(tensor)

        self._cast(cast_function)


class VrocRegistration(LoggerMixin):
    DEFAULT_REGISTRATION_PARAMETERS = {
        "iterations": 800,
        "tau": 2.25,
        "tau_level_decay": 0.0,
        "tau_iteration_decay": 0.0,
        "sigma_x": 1.25,
        "sigma_y": 1.25,
        "sigma_z": 1.25,
        "sigma_level_decay": 0.0,
        "sigma_iteration_decay": 0.0,
        "n_levels": 3,
        "largest_scale_factor": 1.0,
    }

    def __init__(
        self,
        device: TorchDevice = "cuda",
    ):
        self.device = device

    @property
    def available_registration_parameters(self) -> Tuple[str, ...]:
        return tuple(VrocRegistration.DEFAULT_REGISTRATION_PARAMETERS.keys())

    def _get_parameter_value(self, parameters: List[dict], parameter_name: str):
        """Returns the value for the parameter with name parameter_name. First
        found, first returned.

        :param parameters:
        :type parameters: List[dict]
        :param parameter_name:
        :type parameter_name: str
        :return:
        :rtype:
        """

        not_found = object()

        for _parameters in parameters:
            if (value := _parameters.get(parameter_name, not_found)) is not not_found:
                return value

    def _segment_roi(self, image: np.ndarray) -> np.ndarray:
        pass

    def _clip_images(
        self, images: Sequence[ArrayOrTensor], lower: Number, upper: Number
    ):
        return tuple(image.clip(lower, upper) for image in images)

    @timing()
    @convert("debug_output_folder", converter=Path)
    def register(
        self,
        moving_image: ArrayOrTensor,
        fixed_image: ArrayOrTensor,
        moving_mask: ArrayOrTensor | None = None,
        fixed_mask: ArrayOrTensor | None = None,
        moving_landmarks: ArrayOrTensor | None = None,
        fixed_landmarks: ArrayOrTensor | None = None,
        use_masks: MaybeSequence[bool] = True,
        image_spacing: FloatTuple2D | FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: ArrayOrTensor | None = None,
        register_affine: bool = True,
        affine_loss_function: Callable | None = None,
        affine_iterations: int = 300,
        affine_step_size: float = 1e-3,
        affine_enable_translation: bool = True,
        affine_enable_scaling: bool = True,
        affine_enable_rotation: bool = True,
        affine_enable_shearing: bool = True,
        force_type: Literal["demons", "ncc", "ngf"] = "demons",
        gradient_type: Literal["active", "passive", "dual"] = "dual",
        valid_value_range: Tuple[Number, Number] | None = None,
        early_stopping_delta: float = 0.0,
        early_stopping_window: int | None = None,
        default_parameters: dict | None = None,
        default_voxel_value: Number = 0.0,
        return_as_tensor: bool = False,
        debug: bool = False,
        debug_output_folder: PathLike | None = None,
        debug_step_size: int = 10,
        yield_each_step: bool = False,
        mode: str = "standard",
    ) -> RegistrationResult:
        # typing for converted args/kwargs
        debug_output_folder: Path
        if debug_output_folder:
            debug_output_folder.mkdir(exist_ok=True, parents=True)

        # n_spatial_dims is defined by length of image_spacing
        n_spatial_dims = len(image_spacing)
        # n_total_dims = n_spatial_dims + batch dim + color dim
        n_total_dims = n_spatial_dims + 2
        if n_spatial_dims not in {2, 3}:
            raise NotImplementedError(
                "Registration is currently only implemented for 2D and 3D images"
            )

        # check if shapes match
        if moving_image.shape != fixed_image.shape:
            raise ValueError(
                f"Shape mismatch between "
                f"{moving_image.shape=} and {fixed_image.shape=}"
            )

        self.logger.info(f"Got images with shape {moving_image.shape}")

        # cast to torch tensors if inputs are not torch tensors
        # add batch and color dimension and move to specified device if needed
        moving_image = as_tensor(
            moving_image, n_dim=n_total_dims, dtype=torch.float32, device=self.device
        )
        fixed_image = as_tensor(
            fixed_image, n_dim=n_total_dims, dtype=torch.float32, device=self.device
        )
        moving_mask = as_tensor(
            moving_mask, n_dim=n_total_dims, dtype=torch.bool, device=self.device
        )
        fixed_mask = as_tensor(
            fixed_mask, n_dim=n_total_dims, dtype=torch.bool, device=self.device
        )

        if valid_value_range:
            moving_image, fixed_image = self._clip_images(
                images=(moving_image, fixed_image),
                lower=valid_value_range[0],
                upper=valid_value_range[1],
            )
            self.logger.info(
                f"Clip image values to given value range {valid_value_range}"
            )
        else:
            # we set valid value range to (min, max) of the image value range and use
            # min value as default value for resampling (spatial transformer)
            valid_value_range = (
                min(moving_image.min(), fixed_image.min()),
                max(moving_image.max(), fixed_image.max()),
            )

        if initial_vector_field is not None and register_affine:
            raise RuntimeError(
                "Combination of initial_vector_field and register_affine "
                "is not supported yet"
            )

        initial_vector_field = as_tensor(
            initial_vector_field,
            n_dim=n_total_dims,
            dtype=torch.float32,
            device=self.device,
        )

        if register_affine:
            (
                warped_affine_moving_image,
                initial_vector_field,
            ) = run_affine_registration(
                moving_image=moving_image,
                fixed_image=fixed_image,
                moving_mask=moving_mask,
                fixed_mask=fixed_mask,
                loss_function=affine_loss_function,
                n_iterations=affine_iterations,
                step_size=affine_step_size,
                enable_translation=affine_enable_translation,
                enable_scaling=affine_enable_scaling,
                enable_rotation=affine_enable_rotation,
                enable_shearing=affine_enable_shearing,
                default_voxel_value=default_voxel_value,
            )

        # handle ROIs
        if moving_mask is None:
            moving_mask = torch.ones_like(moving_image, dtype=torch.bool)

        if fixed_mask is None:
            fixed_mask = torch.ones_like(fixed_image, dtype=torch.bool)

        # gather all parameters
        # (try passed parameters,
        # then VROC default parameters)
        parameters = {
            param_name: self._get_parameter_value(
                [
                    default_parameters,
                    VrocRegistration.DEFAULT_REGISTRATION_PARAMETERS,
                ],
                param_name,
            )
            for param_name in self.available_registration_parameters
        }
        # regularization_sigma is dependent on spatial dimension of the images
        # 2D: (x, y), 3D: (x, y, z)
        regularization_sigma = (
            parameters["sigma_x"],
            parameters["sigma_y"],
            parameters["sigma_z"],
        )[:n_spatial_dims]

        # delete sigma_z so that it is not logged in the following logger call
        if n_spatial_dims == 2:
            del parameters["sigma_z"]
        self.logger.info(f"Start registration with parameters {parameters}")

        # run VarReg
        scale_factors = tuple(
            parameters["largest_scale_factor"] / 2**i_level
            for i_level in reversed(range(parameters["n_levels"]))
        )
        self.logger.debug(f"Using image pyramid scale factors: {scale_factors}")

        if mode == "standard":
            varreg_class = VariationalRegistration
        else:
            raise NotImplementedError(mode)

        self.logger.info(f"Using the following VarReg class: {varreg_class.__name__}")

        varreg = varreg_class(
            iterations=parameters["iterations"],
            scale_factors=scale_factors,
            use_masks=use_masks,
            force_type=force_type,
            gradient_type=gradient_type,
            tau_level_decay=parameters["tau_level_decay"],
            tau_iteration_decay=parameters["tau_iteration_decay"],
            tau=parameters["tau"],
            regularization_sigma=regularization_sigma,
            sigma_level_decay=parameters["sigma_level_decay"],
            sigma_iteration_decay=parameters["sigma_iteration_decay"],
            restrict_to_mask_bbox=True,
            early_stopping_delta=early_stopping_delta,
            early_stopping_window=early_stopping_window,
            debug=debug,
            debug_output_folder=debug_output_folder,
            debug_step_size=debug_step_size,
            default_voxel_value=default_voxel_value,
        ).to(self.device)

        with torch.autocast(device_type="cuda", enabled=True), torch.inference_mode():
            varreg_result = varreg.run_registration(
                moving_image=moving_image,
                fixed_image=fixed_image,
                moving_mask=moving_mask,
                fixed_mask=fixed_mask,
                moving_landmarks=moving_landmarks,
                fixed_landmarks=fixed_landmarks,
                original_image_spacing=image_spacing,
                initial_vector_field=initial_vector_field,
                yield_each_step=yield_each_step,
            )

        if yield_each_step:

            def yield_registration_steps():
                for step in varreg_result:
                    if step["type"] == "final":
                        step = RegistrationResult(
                            moving_image=moving_image,
                            warped_moving_image=step["warped_moving_image"],
                            warped_affine_moving_image=step[
                                "warped_affine_moving_image"
                            ],
                            fixed_image=fixed_image,
                            moving_mask=moving_mask,
                            warped_moving_mask=step["warped_moving_mask"],
                            warped_affine_moving_mask=step["warped_affine_moving_mask"],
                            fixed_mask=fixed_mask,
                            composed_vector_field=step["composed_vector_field"],
                            vector_fields=step["vector_fields"],
                            debug_info=step["debug_info"],
                        )

                        if not return_as_tensor:
                            step.to_numpy()

                    yield step

            result = yield_registration_steps()

        else:
            result = RegistrationResult(
                moving_image=moving_image,
                warped_moving_image=varreg_result["warped_moving_image"],
                warped_affine_moving_image=varreg_result["warped_affine_moving_image"],
                fixed_image=fixed_image,
                moving_mask=moving_mask,
                warped_moving_mask=varreg_result["warped_moving_mask"],
                warped_affine_moving_mask=varreg_result["warped_affine_moving_mask"],
                fixed_mask=fixed_mask,
                composed_vector_field=varreg_result["composed_vector_field"],
                vector_fields=varreg_result["vector_fields"],
                debug_info=varreg_result["debug_info"],
            )

            if not return_as_tensor:
                result.to_numpy()

        return result
