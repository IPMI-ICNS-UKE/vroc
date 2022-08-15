from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
import torch

from vroc.common_types import Image, Number
from vroc.decorators import timing
from vroc.guesser import ParameterGuesser
from vroc.logger import LoggerMixin
from vroc.metrics import root_mean_squared_error
from vroc.models import VarReg3d
from vroc.preprocessing import affine_registration


@dataclass
class RegistrationResult:
    warped_moving_image: np.ndarray
    composed_vector_field: np.ndarray
    vector_fields: List[np.ndarray]
    warped_affine_moving_image: np.ndarray | None = None
    debug_info: dict | None = None


class ImageWrapper:
    def __init__(self, image: Image):
        if isinstance(image, np.ndarray):
            self._numpy_image = image
            self._sitk_image = None
        elif isinstance(image, sitk.Image):
            self._sitk_image = image
            self._numpy_image = None
        else:
            raise ValueError

    def _sitk_from_numpy(self, image: np.ndarray) -> sitk.Image:
        image = np.swapaxes(image, 0, 2)

        return sitk.GetImageFromArray(image)

    def _numpy_from_sitk(self, image: sitk.Image) -> np.ndarray:
        image = sitk.GetArrayFromImage(image)
        image = np.swapaxes(image, 0, 2)

        return image

    def as_numpy(self) -> np.ndarray:
        if self._numpy_image is None:
            return self._numpy_from_sitk(self._sitk_image)
        else:
            return self._numpy_image

    def as_sitk(self) -> sitk.Image:
        if self._sitk_image is None:
            if (dtype := self._numpy_image.dtype) == bool:
                # sitk cannot handle bool, convert to uint8
                dtype = np.uint8
            return self._sitk_from_numpy(np.asarray(self._numpy_image, dtype=dtype))
        else:
            return self._sitk_image

    @property
    def shape(self):
        if self._numpy_image is not None:
            return self._numpy_image.shape
        else:
            return self._sitk_image.GetSize()

    @property
    def image_spacing(self) -> Tuple:
        return self.as_sitk().GetSpacing()


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

    def _convert_sitk_to_numpy(self, image: sitk.Image) -> Tuple[np.ndarray, dict]:
        meta = {
            "spacing": image.GetSpacing(),
            "size": image.GetSize(),
            "origin": image.GetOrigin(),
            "direction": image.GetDirection(),
            "dtype": image.GetPixelIDValue(),
        }
        image = sitk.GetArrayFromImage(image)
        image = np.swapaxes(image, 0, 2)

        return image, meta

    def _segment_roi(self, image: np.ndarray) -> np.ndarray:
        pass

    def _warp_sitk_image(
        self, image: sitk.Image, transform: sitk.Transform, is_mask: bool = False
    ):
        iterpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
        return sitk.Resample(
            image,
            image,
            transform,
            iterpolator,
            0,
            image.GetPixelID(),
        )

    def _clip_image(self, *images, lower: Number, upper: Number):
        f = sitk.ClampImageFilter()
        f.SetLowerBound(lower)
        f.SetUpperBound(upper)

        return tuple(f.Execute(image) for image in images)

    @staticmethod
    @timing()
    def register_affine(
        moving_image: ImageWrapper,
        fixed_image: ImageWrapper,
        moving_mask: ImageWrapper | None = None,
        fixed_mask: ImageWrapper | None = None,
    ):
        moving_image = moving_image.as_sitk()
        fixed_image = fixed_image.as_sitk()
        moving_mask = moving_mask.as_sitk()
        fixed_mask = fixed_mask.as_sitk()

        _, affine_transform = affine_registration(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
        )

        f = sitk.TransformToDisplacementFieldFilter()
        f.SetSize(fixed_image.GetSize())
        f.SetOutputSpacing((1.0, 1.0, 1.0))
        affine_transform_vector_field = f.Execute(affine_transform)
        affine_transform_vector_field.SetDirection(fixed_image.GetDirection())
        affine_transform_vector_field = sitk.GetArrayFromImage(
            affine_transform_vector_field
        )
        affine_transform_vector_field = affine_transform_vector_field.astype(np.float32)
        affine_transform_vector_field /= fixed_image.GetSpacing()
        affine_transform_vector_field = np.swapaxes(affine_transform_vector_field, 0, 2)
        # move displacement axis to the start
        affine_transform_vector_field = np.moveaxis(
            affine_transform_vector_field, -1, 0
        )

        return affine_transform_vector_field

    @timing()
    def register(
        self,
        moving_image: Image,
        fixed_image: Image,
        moving_mask: Image | None = None,
        fixed_mask: Image | None = None,
        register_affine: bool = True,
        segment_roi: bool = True,
        valid_value_range: Tuple[Number, Number] | None = None,
        early_stopping_delta: float = 0.0,
        early_stopping_window: int = 20,
        default_parameters: dict | None = None,
        debug: bool = False,
    ) -> RegistrationResult:
        # these are default registration parameters used if no parameter guesser is
        # passed or the given parameter guesser returns only a subset of parameters
        self.default_parameters = (
            default_parameters or VrocRegistration.DEFAULT_REGISTRATION_PARAMETERS
        )

        moving_image = ImageWrapper(moving_image)
        fixed_image = ImageWrapper(fixed_image)

        if valid_value_range:
            moving_image, fixed_image = self._clip_image(
                moving_image.as_sitk(),
                fixed_image.as_sitk(),
                lower=valid_value_range[0],
                upper=valid_value_range[1],
            )
            self.logger.info(f"Clip image values to {valid_value_range}")

            moving_image = ImageWrapper(moving_image)
            fixed_image = ImageWrapper(fixed_image)

        if moving_mask is not None:
            moving_mask = ImageWrapper(moving_mask)
        else:
            moving_mask = None

        if fixed_mask is not None:
            fixed_mask = ImageWrapper(fixed_mask)
        else:
            fixed_mask = None

        if register_affine:
            affine_transform_vector_field = VrocRegistration.register_affine(
                moving_image=moving_image,
                fixed_image=fixed_image,
                moving_mask=moving_mask,
                fixed_mask=fixed_mask,
            )

            # # set warped image as new moving image
            # warped_affine_moving_image = ImageWrapper(warped_affine_moving_image)
            # if moving_mask is not None:
            #     warped_mask = ImageWrapper(
            #         self._warp_sitk_image(
            #             image=moving_mask.as_sitk(),
            #             transform=affine_transform,
            #             is_mask=True,
            #         )
            #     )

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

        # handle ROI
        if moving_mask is None and fixed_mask is None:
            if segment_roi:
                moving_mask = self._segment_roi(moving_image.as_numpy())
                fixed_mask = self._segment_roi(fixed_image.as_numpy())
            else:
                moving_mask = np.ones(moving_image.shape, dtype=np.uint8)
                fixed_mask = np.ones(fixed_image.shape, dtype=np.uint8)

        elif moving_mask is not None and fixed_mask is not None:
            # use passed masks
            moving_mask = moving_mask.as_numpy()
            fixed_mask = fixed_mask.as_numpy()
        else:
            raise RuntimeError("Please pass both masks XOR use ROI segmenter")

        image_spacing = fixed_image.image_spacing
        fixed_image = fixed_image.as_numpy()
        moving_image = moving_image.as_numpy()

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

        if register_affine:
            affine_transform_vector_field = torch.as_tensor(
                affine_transform_vector_field[np.newaxis]
            ).to(self.device)
        else:
            affine_transform_vector_field = None

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
