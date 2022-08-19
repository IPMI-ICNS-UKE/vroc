from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch

from vroc.affine import run_affine_registration
from vroc.common_types import FloatTuple3D, Image, Number
from vroc.decorators import timing
from vroc.guesser import ParameterGuesser
from vroc.logger import LoggerMixin
from vroc.metrics import root_mean_squared_error
from vroc.models import VarReg3d


@dataclass
class RegistrationResult:
    warped_moving_image: np.ndarray
    composed_vector_field: np.ndarray
    vector_fields: List[np.ndarray]
    warped_affine_moving_image: np.ndarray | None = None
    debug_info: dict | None = None


class VrocRegistration(LoggerMixin):
    DEFAULT_REGISTRATION_PARAMETERS = {
        "iterations": 1000,
        "tau": 2.0,
        "sigma_x": 1.25,
        "sigma_y": 1.25,
        "sigma_z": 1.25,
        "n_levels": 3,
    }

    def __init__(
        self,
        roi_segmenter,
        feature_extractor: "FeatureExtractor" | None = None,
        parameter_guesser: ParameterGuesser | None = None,
        device: str = "cuda",
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
        moving_image: np.ndarray,
        fixed_image: np.ndarray,
        moving_mask: np.ndarray | None = None,
        fixed_mask: np.ndarray | None = None,
        image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        register_affine: bool = True,
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

        # add batch and color dimension and move to specified device
        moving_image = torch.as_tensor(moving_image[np.newaxis, np.newaxis]).to(
            self.device
        )
        fixed_image = torch.as_tensor(fixed_image[np.newaxis, np.newaxis]).to(
            self.device
        )
        moving_mask = torch.as_tensor(moving_mask[np.newaxis, np.newaxis]).to(
            self.device
        )
        fixed_mask = torch.as_tensor(fixed_mask[np.newaxis, np.newaxis]).to(self.device)

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
                n_iterations=300,
            )
        else:
            affine_transform_vector_field = None

            # if debug:
            #     # calculate RMSE before and after affine registration
            #     before = root_mean_squared_error(
            #         image=moving_image.as_numpy(),
            #         reference_image=fixed_image.as_numpy(),
            #         mask=fixed_mask.as_numpy(),
            #     )
            #     after = root_mean_squared_error(
            #         image=warped_affine_moving_image.as_numpy(),
            #         reference_image=fixed_image.as_numpy(),
            #         mask=fixed_mask.as_numpy(),
            #     )
            #
            #     self.logger.debug(
            #         f"Affine registration: RMSE before: {before}, RMSE after {after}"
            #     )

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
            variant="demons",
            forces="dual",
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
