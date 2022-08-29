from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch

from vroc.affine import run_affine_registration
from vroc.common_types import FloatTuple3D, Image, Number, TorchDevice
from vroc.convert import as_tensor
from vroc.decorators import timing
from vroc.guesser import ParameterGuesser
from vroc.logger import LoggerMixin
from vroc.metrics import root_mean_squared_error
from vroc.models import DemonsVectorFieldBooster, VarReg3d


@dataclass
class RegistrationResult:
    warped_moving_image: np.ndarray
    composed_vector_field: np.ndarray
    vector_fields: List[np.ndarray]
    warped_affine_moving_image: np.ndarray | None = None
    debug_info: dict | None = None


class VrocRegistration(LoggerMixin):
    DEFAULT_REGISTRATION_PARAMETERS = {
        "iterations": 800,
        "tau": 2.25,
        "sigma_x": 1.25,
        "sigma_y": 1.25,
        "sigma_z": 1.25,
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

        # gather all parameters (try guessed parameters, then default parameters)
        parameters = {
            param_name: self._get_parameter_value(
                [guessed_parameters, default_parameters], param_name
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
            tau=parameters["tau"],
            regularization_sigma=(
                parameters["sigma_x"],
                parameters["sigma_y"],
                parameters["sigma_z"],
            ),
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

            warped_moving_image = (
                varreg_result["warped_moving_image"].cpu().numpy().squeeze(axis=(0, 1))
            )
            composed_vector_field = (
                varreg_result["composed_vector_field"].cpu().numpy().squeeze(axis=0)
            )
            vector_fields = [
                vector_field.cpu().numpy().squeeze(axis=0)
                for vector_field in varreg_result["vector_fields"]
            ]

        result = RegistrationResult(
            warped_moving_image=warped_moving_image,
            composed_vector_field=composed_vector_field,
            vector_fields=vector_fields,
            debug_info=varreg_result["debug_info"],
        )

        return result
