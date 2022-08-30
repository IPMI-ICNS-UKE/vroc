from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.optim as optim

from vroc.affine import run_affine_registration
from vroc.common_types import FloatTuple3D, Image, Number, TorchDevice
from vroc.convert import as_tensor
from vroc.decorators import timing
from vroc.guesser import ParameterGuesser
from vroc.logger import LoggerMixin
from vroc.loss import TRELoss
from vroc.metrics import root_mean_squared_error
from vroc.models import DemonsVectorFieldBooster, VarReg3d


@dataclass
class RegistrationResult:
    # initial images
    moving_image: np.ndarray | torch.Tensor
    fixed_image: np.ndarray | torch.Tensor

    warped_moving_image: np.ndarray | torch.Tensor
    composed_vector_field: np.ndarray | torch.Tensor
    vector_fields: List[np.ndarray | torch.Tensor]

    # initial masks
    moving_mask: np.ndarray | torch.Tensor | None = None
    fixed_mask: np.ndarray | torch.Tensor | None = None

    warped_affine_moving_image: np.ndarray | torch.Tensor | None = None
    debug_info: dict | None = None

    def to_numpy(self):
        def cast(
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

        self.moving_image = cast(self.moving_image)
        self.fixed_image = cast(self.fixed_image)

        self.warped_moving_image = cast(self.warped_moving_image)
        self.composed_vector_field = cast(self.composed_vector_field)
        self.vector_fields = cast(self.vector_fields)

        self.moving_mask = cast(self.moving_mask)
        self.fixed_mask = cast(self.fixed_mask)

        self.warped_affine_moving_image = cast(self.warped_affine_moving_image)


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
    }

    def __init__(
        self,
        roi_segmenter=None,
        feature_extractor: "FeatureExtractor" | None = None,
        parameter_guesser: ParameterGuesser | None = None,
        device: TorchDevice = "cuda",
    ):
        self.roi_segmenter = roi_segmenter

        self.feature_extractor = feature_extractor
        self.parameter_guesser = parameter_guesser

        # if parameter_guesser is set we need a feature_extractor
        if self.parameter_guesser and not self.feature_extractor:
            raise ValueError(
                "Feature extractor can not be None if a parameter guesser is passed"
            )

        self.device = device

    @property
    def available_registration_parameters(self) -> Tuple[str]:
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
        self, images: Sequence[np.ndarray | torch.Tensor], lower: Number, upper: Number
    ):
        return tuple(image.clip(lower, upper) for image in images)

    @timing()
    def register(
        self,
        moving_image: np.ndarray | torch.Tensor,
        fixed_image: np.ndarray | torch.Tensor,
        moving_mask: np.ndarray | torch.Tensor | None = None,
        fixed_mask: np.ndarray | torch.Tensor | None = None,
        image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        register_affine: bool = True,
        affine_loss_fn: Callable | None = None,
        force_type: Literal["demons", "ncc", "ngf"] = "demons",
        gradient_type: Literal["active", "passive", "dual"] = "dual",
        segment_roi: bool = True,
        valid_value_range: Tuple[Number, Number] | None = None,
        early_stopping_delta: float = 0.0,
        early_stopping_window: int | None = None,
        default_parameters: dict | None = None,
        return_as_tensor: bool = False,
        debug: bool = False,
    ) -> RegistrationResult:
        # these are default registration parameters used if no parameter guesser is
        # passed or the given parameter guesser returns only a subset of parameters
        self.default_parameters = (
            default_parameters or VrocRegistration.DEFAULT_REGISTRATION_PARAMETERS
        )

        # n_spatial_dims is defined by length of image_spacing
        n_spatial_dims = len(image_spacing)
        if n_spatial_dims not in (3,):
            raise NotImplementedError(
                "Registration is currently only implemented for volumetric (3D) images"
            )

        # cast to torch tensors if inputs are not torch tensors
        # add batch and color dimension and move to specified device if needed
        moving_image = as_tensor(
            moving_image, n_dim=5, dtype=torch.float32, device=self.device
        )
        fixed_image = as_tensor(
            fixed_image, n_dim=5, dtype=torch.float32, device=self.device
        )
        moving_mask = as_tensor(
            moving_mask, n_dim=5, dtype=torch.bool, device=self.device
        )
        fixed_mask = as_tensor(
            fixed_mask, n_dim=5, dtype=torch.bool, device=self.device
        )

        if valid_value_range:
            moving_image, fixed_image = self._clip_images(
                images=(moving_image, fixed_image),
                lower=valid_value_range[0],
                upper=valid_value_range[1],
            )
            self.logger.info(f"Clip image values to {valid_value_range}")

        if register_affine:
            (
                warped_affine_moving_image,
                affine_transform_vector_field,
            ) = run_affine_registration(
                moving_image=moving_image,
                fixed_image=fixed_image,
                moving_mask=moving_mask,
                fixed_mask=fixed_mask,
                loss_function=affine_loss_fn,
                n_iterations=300,
            )
        else:
            affine_transform_vector_field = None

        # handle ROIs
        # passed masks overwrite ROI segmenter
        if moving_mask is None and segment_roi:
            moving_mask = self._segment_roi(moving_image)
        elif moving_mask is None and not segment_roi:
            moving_mask = torch.ones_like(moving_image, dtype=torch.bool)

        if fixed_mask is None and segment_roi:
            fixed_mask = self._segment_roi(fixed_image)
        elif fixed_mask is None and not segment_roi:
            fixed_mask = torch.ones_like(fixed_image, dtype=torch.bool)

        if self.feature_extractor and self.parameter_guesser:
            # feature extraction
            features = self.feature_extractor.extract(
                fixed_image=fixed_image,
                moving_image=moving_image,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                image_spacing=image_spacing,
            )
            guessed_parameters = self.parameter_guesser.guess(features)
            self.logger.info(f"Guessed parameters: {guessed_parameters}")
        else:
            self.logger.info(f"No parameters were guessed")
            guessed_parameters = {}

        # gather all parameters
        # (try guessed parameters, then passed default parameters,
        # then VROC default parameters)
        parameters = {
            param_name: self._get_parameter_value(
                [
                    guessed_parameters,
                    default_parameters,
                    VrocRegistration.DEFAULT_REGISTRATION_PARAMETERS,
                ],
                param_name,
            )
            for param_name in self.available_registration_parameters
        }

        self.logger.info(f"Start registration with parameters {parameters}")

        # run VarReg
        scale_factors = tuple(
            1 / 2**i_level for i_level in reversed(range(parameters["n_levels"]))
        )

        varreg = VarReg3d(
            iterations=parameters["iterations"],
            scale_factors=scale_factors,
            force_type=force_type,
            gradient_type=gradient_type,
            tau_level_decay=parameters["tau_level_decay"],
            tau_iteration_decay=parameters["tau_iteration_decay"],
            tau=parameters["tau"],
            regularization_sigma=(
                parameters["sigma_x"],
                parameters["sigma_y"],
                parameters["sigma_z"],
            ),
            sigma_level_decay=parameters["sigma_level_decay"],
            sigma_iteration_decay=parameters["sigma_iteration_decay"],
            restrict_to_mask_bbox=True,
            early_stopping_delta=early_stopping_delta,
            early_stopping_window=early_stopping_window,
            debug=debug,
        ).to(self.device)

        with torch.inference_mode():
            varreg_result = varreg.run_registration(
                moving_image=moving_image,
                fixed_image=fixed_image,
                moving_mask=moving_mask,
                fixed_mask=fixed_mask,
                original_image_spacing=image_spacing,
                initial_vector_field=affine_transform_vector_field,
            )

            # composed_vector_field = varreg_result["composed_vector_field"]
            #
            # booster = DemonsVectorFieldBooster(
            #     shape=composed_vector_field.shape[2:], n_iterations=5
            # )
            # state = torch.load("/datalake/learn2reg/demons_vector_field_booster.pth")
            # booster.load_state_dict(state["model"])
            # booster = booster.to(self.device)
            #
            # boosted = booster(
            #     (moving_image + 1024) / 4095,
            #     (fixed_image + 1024) / 4095,
            #     composed_vector_field,
            #     image_spacing,
            # )
            # varreg_result["composed_vector_field"] = boosted

        result = RegistrationResult(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            warped_moving_image=varreg_result["warped_moving_image"],
            composed_vector_field=varreg_result["composed_vector_field"],
            vector_fields=varreg_result["vector_fields"],
            debug_info=varreg_result["debug_info"],
        )

        if not return_as_tensor:
            result.to_numpy()

        return result

    @timing()
    def register_and_train_boosting(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        n_iterations: int,
        moving_keypoints: torch.Tensor,
        fixed_keypoints: torch.Tensor,
        moving_image: np.ndarray | torch.Tensor,
        fixed_image: np.ndarray | torch.Tensor,
        moving_mask: np.ndarray | torch.Tensor | None = None,
        fixed_mask: np.ndarray | torch.Tensor | None = None,
        image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        register_affine: bool = True,
        affine_loss_fn: Callable | None = None,
        force_type: Literal["demons", "ncc", "ngf"] = "demons",
        gradient_type: Literal["active", "passive", "dual"] = "dual",
        segment_roi: bool = True,
        valid_value_range: Tuple[Number, Number] | None = None,
        early_stopping_delta: float = 0.0,
        early_stopping_window: int | None = None,
        default_parameters: dict | None = None,
        return_as_tensor: bool = False,
        debug: bool = False,
    ) -> RegistrationResult:
        registration_result = self.register(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            image_spacing=image_spacing,
            register_affine=register_affine,
            affine_loss_fn=affine_loss_fn,
            force_type=force_type,
            gradient_type=gradient_type,
            segment_roi=segment_roi,
            valid_value_range=valid_value_range,
            early_stopping_delta=early_stopping_delta,
            early_stopping_window=early_stopping_window,
            default_parameters=default_parameters,
            return_as_tensor=True,  # we need tensors for training
            debug=debug,
        )

        model = model.to(self.device)

        moving_image = torch.clone(registration_result.warped_moving_image)
        fixed_image = torch.clone(registration_result.fixed_image)

        moving_image = (moving_image - valid_value_range[0]) / (
            valid_value_range[1] - valid_value_range[0]
        )
        fixed_image = (fixed_image - valid_value_range[0]) / (
            valid_value_range[1] - valid_value_range[0]
        )

        moving_image = torch.clip(moving_image, 0, 1)
        fixed_image = torch.clip(fixed_image, 0, 1)

        moving_mask = torch.clone(registration_result.moving_mask)
        fixed_mask = torch.clone(registration_result.fixed_mask)

        composed_vector_field = torch.clone(registration_result.composed_vector_field)
        image_spacing = torch.as_tensor(image_spacing, device=self.device)

        tre_loss = TRELoss(apply_sqrt=False, reduction=None)
        gradient_scaler = torch.cuda.amp.GradScaler()

        tre_loss_before_boosting = tre_loss(
            composed_vector_field,
            moving_keypoints,
            fixed_keypoints,
            image_spacing,
        )
        tre_metric_before_boosting = tre_loss_before_boosting.sqrt().mean()
        max_iteration_length = len(str(n_iterations))
        for i_iteration in range(n_iterations):

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                vector_field_boost = model(
                    moving_image,
                    fixed_image,
                    moving_mask,
                    fixed_mask,
                    composed_vector_field,
                    image_spacing,
                )

                # compose initial vector field with vector field boost
                composed_boosted_vector_field = (
                    vector_field_boost
                    + model.spatial_transformer(
                        composed_vector_field, vector_field_boost
                    )
                )

                tre_loss_after_boosting = tre_loss(
                    composed_boosted_vector_field,
                    moving_keypoints,
                    fixed_keypoints,
                    image_spacing,
                )

                tre_metric_after_boosting = tre_loss_after_boosting.sqrt().mean()

                # loss = (tre_loss_after_boosting - tre_loss_before_boosting).mean()
                # # penalize worsening of TRE more
                # if loss > 0:
                #     loss = loss**2

                loss = tre_metric_after_boosting

            gradient_scaler.scale(loss).backward()
            gradient_scaler.step(optimizer)
            gradient_scaler.update()

            self.logger.info(
                f"Train boosting, iteration {i_iteration:0{max_iteration_length}d} / "
                f"loss: {loss:.6f} / "
                f"TRE before: {tre_metric_before_boosting} / "
                f"TRE after: {tre_metric_after_boosting}"
            )

        registration_result.composed_vector_field = composed_boosted_vector_field
        # warp moving image with composed boosed vector field
        warped_moving_image = model.spatial_transformer(
            moving_image, composed_boosted_vector_field
        )
        registration_result.warped_moving_image = warped_moving_image
        registration_result.vector_fields.append(vector_field_boost)

        if not return_as_tensor:
            registration_result.to_numpy()

        return registration_result
