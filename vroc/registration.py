from __future__ import annotations

from typing import Tuple

import numpy as np
import SimpleITK as sitk
import torch

from vroc.common_types import Image, Number
from vroc.decorators import timing
from vroc.feature_extractor import FeatureExtractor
from vroc.guesser import ParameterGuesser
from vroc.logger import LoggerMixin
from vroc.metrics import root_mean_squared_error
from vroc.models import TrainableVarRegBlock
from vroc.preprocessing import affine_registration


class ImageWrapper:
    def __init__(self, image: Image):
        if isinstance(image, np.ndarray):
            self._numpy_image = image
        elif isinstance(image, sitk.Image):
            self._sitk_image = image
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
        try:
            return self._numpy_image
        except AttributeError:
            return self._numpy_from_sitk(self._sitk_image)

    def as_sitk(self) -> sitk.Image:
        try:
            return self._sitk_image
        except AttributeError:
            return self._sitk_from_numpy(self._numpy_image)

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
    def __init__(
        self,
        roi_segmenter,
        feature_extractor: FeatureExtractor,
        parameter_guesser: ParameterGuesser,
        debug: bool = False,
        device: str = "cuda",
    ):
        self.roi_segmenter = roi_segmenter
        self.feature_extractor = feature_extractor
        self.parameter_guesser = parameter_guesser
        self.device = device

        self.debug = debug

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
        moving_image: sitk.Image,
        fixed_image: sitk.Image,
        moving_mask: sitk.Image | None = None,
        fixed_mask: sitk.Image | None = None,
    ):
        return affine_registration(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
        )

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
    ):
        transforms = []

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

            # moving_image = sitk.HistogramMatching(
            #     moving_image, fixed_image, numberOfHistogramLevels=1024,
            #     numberOfMatchPoints=7
            # )
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
            warped_image, affine_transform = VrocRegistration.register_affine(
                moving_image=moving_image.as_sitk(),
                fixed_image=fixed_image.as_sitk(),
                moving_mask=moving_mask.as_sitk(),
                fixed_mask=fixed_mask.as_sitk(),
            )
            transforms.append(affine_transform)
            # set warped image as new moving image
            warped_image = ImageWrapper(warped_image)
            if moving_mask is not None:
                warped_mask = ImageWrapper(
                    self._warp_sitk_image(
                        image=moving_mask.as_sitk(),
                        transform=affine_transform,
                        is_mask=True,
                    )
                )

            if self.debug:
                # calculate RMSE before and after affine registration
                before = root_mean_squared_error(
                    image=moving_image.as_numpy(),
                    reference_image=fixed_image.as_numpy(),
                    mask=fixed_mask.as_numpy(),
                )
                after = root_mean_squared_error(
                    image=warped_image.as_numpy(),
                    reference_image=fixed_image.as_numpy(),
                    mask=fixed_mask.as_numpy(),
                )

                self.logger.debug(
                    f"Affine registration: RMSE before: {before}, RMSE after {after}"
                )

            # set transformed image/mask as new moving image/mask
            moving_image = warped_image
            if moving_mask is not None:
                moving_mask = warped_mask

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

        # feature extraction
        features = self.feature_extractor.extract(
            fixed_image=fixed_image,
            moving_image=moving_image,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            image_spacing=image_spacing,
        )
        parameters = self.parameter_guesser.guess(features)
        self.logger.info(f"Guessed parameters: {parameters}")

        # run varreg
        scale_factors = tuple(
            1 / 2**i_level for i_level in reversed(range(parameters["n_levels"]))
        )

        varreg = TrainableVarRegBlock(
            iterations=parameters["iterations"],
            scale_factors=scale_factors,
            demon_forces="symmetric",
            tau=parameters["tau"],
            regularization_sigma=(
                parameters["sigma_x"],
                parameters["sigma_y"],
                parameters["sigma_z"],
            ),
            restrict_to_mask=True,
        ).to(self.device)

        moving_image = torch.as_tensor(moving_image[np.newaxis, np.newaxis]).to(
            self.device
        )
        fixed_image = torch.as_tensor(fixed_image[np.newaxis, np.newaxis]).to(
            self.device
        )
        fixed_mask = torch.as_tensor(fixed_mask[np.newaxis, np.newaxis]).to(self.device)

        with torch.inference_mode():
            warped_image, vector_field, misc = varreg.forward(
                fixed_image, fixed_mask, moving_image, image_spacing
            )

        warped_image = warped_image.cpu().numpy().squeeze(axis=(0, 1))
        vector_field = vector_field.cpu().numpy().squeeze(axis=0)
        transforms.append(vector_field)

        return warped_image, transforms
