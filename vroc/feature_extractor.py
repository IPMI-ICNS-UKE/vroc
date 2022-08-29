from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from vroc.affine import run_affine_registration
from vroc.common_types import FloatTuple3D, TorchDevice
from vroc.convert import as_tensor
from vroc.interpolation import resize_spacing
from vroc.logger import LoggerMixin
from vroc.models import AutoEncoder, VarReg3d
from vroc.oriented_histogram import OrientedHistogram
from vroc.registration import VrocRegistration


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class OrientedHistogramFeatureExtrator(FeatureExtractor, LoggerMixin):
    DEFAULT_REGISTRATION_PARAMETERS = {
        "iterations": 200,
        "tau": 2.0,
        "sigma_x": 1.0,
        "sigma_y": 1.0,
        "sigma_z": 1.0,
        "n_levels": 1,
    }

    def __init__(
        self,
        n_bins: int = 16,
        registration_parameters: dict | None = None,
        device: str | torch.device = "cuda",
    ):
        self.n_bins = n_bins
        self.registration_parameters = (
            registration_parameters
            or OrientedHistogramFeatureExtrator.DEFAULT_REGISTRATION_PARAMETERS
        )
        self.device = torch.device(device)

    def calculate_oriented_histogram(
        self,
        moving_image: np.ndarray | torch.Tensor,
        fixed_image: np.ndarray | torch.Tensor,
        moving_mask: np.ndarray | torch.Tensor,
        fixed_mask: np.ndarray | torch.Tensor,
        image_spacing: FloatTuple3D,
        target_image_spacing: FloatTuple3D = (4.0, 4.0, 4.0),
        n_bins: int = 16,
        device: TorchDevice = "cuda",
    ):
        self.logger.info(f"Calculating oriented histogram with {n_bins} bins")
        if (n_dims := len(image_spacing)) != len(target_image_spacing):
            raise ValueError(
                f"Dimension mismatch between "
                f"{image_spacing=} and {target_image_spacing=}"
            )
        device = torch.device(device)

        # convert images to 5D tensors, i.e. (1, 1, x_size, y_size, z_size)
        moving_image = as_tensor(
            moving_image, n_dim=n_dims + 2, dtype=torch.float32, device=device
        )
        fixed_image = as_tensor(
            fixed_image, n_dim=n_dims + 2, dtype=torch.float32, device=device
        )
        moving_mask = as_tensor(
            moving_mask, n_dim=n_dims + 2, dtype=torch.bool, device=device
        )
        fixed_mask = as_tensor(
            fixed_mask, n_dim=n_dims + 2, dtype=torch.bool, device=device
        )

        fixed_image = resize_spacing(
            fixed_image,
            input_image_spacing=image_spacing,
            output_image_spacing=target_image_spacing,
        )
        moving_image = resize_spacing(
            moving_image,
            input_image_spacing=image_spacing,
            output_image_spacing=target_image_spacing,
        )
        fixed_mask = resize_spacing(
            fixed_mask,
            input_image_spacing=image_spacing,
            output_image_spacing=target_image_spacing,
            order=0,
        )
        moving_mask = resize_spacing(
            moving_mask,
            input_image_spacing=image_spacing,
            output_image_spacing=target_image_spacing,
            order=0,
        )
        # calculate union mask after resizing moving and fixed mask
        union_mask = fixed_mask | moving_mask

        registration = VrocRegistration(
            roi_segmenter=None,
            feature_extractor=None,
            parameter_guesser=None,
            device=device,
        )

        registration_result_1 = registration.register(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            register_affine=True,
            default_parameters=self.registration_parameters,
            debug=False,
        )
        registration_result_2 = registration.register(
            moving_image=fixed_image,
            fixed_image=moving_image,
            moving_mask=fixed_mask,
            fixed_mask=moving_mask,
            register_affine=True,
            default_parameters=self.registration_parameters,
            debug=False,
        )

        # vector_fields is list [affine_vector_field, varreg_vector_field]
        vector_field_1 = registration_result_1.vector_fields[-1]
        vector_field_2 = registration_result_2.vector_fields[-1]

        oriented_histogram = OrientedHistogram(n_bins=n_bins)
        union_mask = union_mask.detach().cpu().numpy().squeeze(axis=(0, 1))

        oh_1 = oriented_histogram.calculate(vector_field_1, mask=union_mask)
        oh_2 = oriented_histogram.calculate(vector_field_2, mask=union_mask)

        return np.mean(
            (
                oh_1,
                oh_2,
            ),
            axis=0,
        )

    def extract(
        self,
        fixed_image: np.ndarray | torch.Tensor,
        moving_image: np.ndarray | torch.Tensor,
        fixed_mask: np.ndarray | torch.Tensor,
        moving_mask: np.ndarray | torch.Tensor,
        image_spacing: FloatTuple3D,
        target_image_spacing: FloatTuple3D = (4.0, 4.0, 4.0),
    ) -> np.ndarray:
        features = self.calculate_oriented_histogram(
            fixed_image=fixed_image,
            moving_image=moving_image,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            image_spacing=image_spacing,
            target_image_spacing=target_image_spacing,
            n_bins=self.n_bins,
            device=self.device,
        )

        return features

    @property
    def feature_name(self) -> str:
        return f"OH_{self.n_bins}"
