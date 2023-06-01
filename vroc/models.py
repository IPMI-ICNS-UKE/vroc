from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vroc.blocks import (
    DemonForces,
    GaussianSmoothing2d,
    GaussianSmoothing3d,
    SpatialTransformer,
)
from vroc.checks import are_of_same_length, is_tuple, is_tuple_of_tuples
from vroc.common_types import (
    FloatTuple2D,
    FloatTuple3D,
    IntTuple2D,
    IntTuple3D,
    MaybeSequence,
    Number,
)
from vroc.decay import exponential_decay
from vroc.decorators import timing
from vroc.helper import get_bounding_box
from vroc.interpolation import match_vector_field, rescale
from vroc.logger import LoggerMixin
from vroc.loss import TRELoss


class BaseIterativeRegistration(ABC, nn.Module, LoggerMixin):
    def _create_spatial_transformers(self, image_shape: Tuple[int, ...], device):
        if not image_shape == self._image_shape:
            self._image_shape = image_shape
            self._full_size_spatial_transformer = SpatialTransformer(
                shape=image_shape, default_value=self.default_voxel_value
            ).to(device)

            for i_level, scale_factor in enumerate(self.scale_factors):
                scaled_image_shape = tuple(
                    int(round(s * scale_factor)) for s in image_shape
                )

                try:
                    module = getattr(self, f"spatial_transformer_level_{i_level}")
                    del module
                except AttributeError:
                    pass
                self.add_module(
                    name=f"spatial_transformer_level_{i_level}",
                    module=SpatialTransformer(
                        shape=scaled_image_shape, default_value=self.default_voxel_value
                    ).to(device),
                )

    def _perform_scaling(
        self, *images, scale_factor: float = 1.0
    ) -> List[torch.Tensor]:
        scaled = []
        for image in images:
            if image is not None:
                is_mask = image.dtype == torch.bool
                order = 0 if is_mask else 1

                image = rescale(image, factor=scale_factor, order=order)

            scaled.append(image)

        return scaled

    def _calculate_metric(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        fixed_mask: torch.Tensor | None = None,
    ) -> float:
        if fixed_mask is not None:
            fixed_image = fixed_image[fixed_mask]
            moving_image = moving_image[fixed_mask]

        return float(F.mse_loss(fixed_image, moving_image))

    def _check_early_stopping(self, metrics: List[float], i_level: int) -> bool:
        early_stop = False
        if (
            self.early_stopping_delta[i_level]
            and self.early_stopping_window[i_level]
            and len(metrics) >= self.early_stopping_window[i_level]
        ):
            window = np.array(metrics[-self.early_stopping_window[i_level] :])
            window_rel_changes = 1 - window[1:] / window[:-1]

            mean_rel_change = window_rel_changes.mean()

            self.logger.debug(
                f"Mean relative metric change over window of "
                f"{self.early_stopping_window[i_level]} steps is {mean_rel_change:.6f}"
            )

            if mean_rel_change < self.early_stopping_delta[i_level]:
                early_stop = True
                self.logger.debug(
                    f"Early stopping triggered for level {i_level} after iteration "
                    f"{len(metrics)}: {mean_rel_change:.6f} < "
                    f"{self.early_stopping_delta[i_level]:.6f}"
                )

        return early_stop

    @staticmethod
    def _expand_to_level_tuple(
        value: Any, n_levels: int, is_tuple: bool = False, expand_none: bool = False
    ) -> Optional[Tuple]:
        if not expand_none and value is None:
            return value
        else:
            if not is_tuple:
                return (value,) * n_levels
            elif is_tuple and not is_tuple_of_tuples(value):
                return (value,) * n_levels
            else:
                return value

    def _update_step(
        self,
        level: int,
        iteration: int,
        scale_factor: float,
        vector_field: torch.Tensor,
        moving_image: torch.Tensor,
        warped_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def run_registration(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
    ):
        raise NotImplementedError

    @timing()
    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
    ):
        return self.run_registration(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            original_image_spacing=original_image_spacing,
            initial_vector_field=initial_vector_field,
        )


class VariationalRegistration(BaseIterativeRegistration):
    _GAUSSIAN_SMOOTHING = {2: GaussianSmoothing2d, 3: GaussianSmoothing3d}

    def __init__(
        self,
        scale_factors: MaybeSequence[float] = (1.0,),
        use_masks: MaybeSequence[bool] = True,
        iterations: MaybeSequence[int] = 100,
        tau: MaybeSequence[float] = 1.0,
        tau_level_decay: float = 0.0,
        tau_iteration_decay: float = 0.0,
        force_type: Literal["demons"] = "demons",
        gradient_type: Literal["active", "passive", "dual"] = "dual",
        regularization_sigma: MaybeSequence[FloatTuple2D | FloatTuple3D] = (
            1.0,
            1.0,
            1.0,
        ),
        regularization_radius: MaybeSequence[IntTuple2D | IntTuple3D] | None = None,
        sigma_level_decay: float = 0.0,
        sigma_iteration_decay: float = 0.0,
        original_image_spacing: FloatTuple2D | FloatTuple3D = (1.0, 1.0, 1.0),
        use_image_spacing: bool = False,
        restrict_to_mask_bbox: bool = False,
        early_stopping_delta: float = 0.0,
        early_stopping_window: int | None = 20,
        boosting_model: nn.Module | None = None,
        default_voxel_value: Number = 0.0,
        debug: bool = False,
        debug_output_folder: Path | None = None,
        debug_step_size: int = 10,
    ):
        super().__init__()

        n_spatial_dims = len(original_image_spacing)

        if not is_tuple(scale_factors, min_length=1):
            scale_factors = (scale_factors,)
        self.scale_factors = scale_factors

        self.scale_factors = scale_factors  # this also defines "n_levels"
        self.use_masks = VariationalRegistration._expand_to_level_tuple(
            use_masks, n_levels=self.n_levels
        )
        self.iterations = VariationalRegistration._expand_to_level_tuple(
            iterations, n_levels=self.n_levels
        )

        self.tau = VariationalRegistration._expand_to_level_tuple(
            tau, n_levels=self.n_levels
        )
        self.tau_level_decay = tau_level_decay
        self.tau_iteration_decay = tau_iteration_decay

        self.regularization_sigma = VariationalRegistration._expand_to_level_tuple(
            regularization_sigma, n_levels=self.n_levels, is_tuple=True
        )
        self.regularization_radius = VariationalRegistration._expand_to_level_tuple(
            regularization_radius, n_levels=self.n_levels, is_tuple=True
        )
        self.sigma_level_decay = sigma_level_decay
        self.sigma_iteration_decay = sigma_iteration_decay
        self.original_image_spacing = original_image_spacing
        self.use_image_spacing = use_image_spacing

        self.forces = gradient_type

        self.restrict_to_mask_bbox = restrict_to_mask_bbox
        self.early_stopping_delta = VariationalRegistration._expand_to_level_tuple(
            early_stopping_delta, n_levels=self.n_levels
        )
        self.early_stopping_window = VariationalRegistration._expand_to_level_tuple(
            early_stopping_window, n_levels=self.n_levels, expand_none=True
        )
        self.boosting_model = boosting_model
        self.default_voxel_value = default_voxel_value
        self.debug = debug
        self.debug_output_folder = debug_output_folder
        self.debug_step_size = debug_step_size

        if self.debug:
            from vroc.plot import RegistrationProgressPlotter

            self._plotter = RegistrationProgressPlotter(
                output_folder=self.debug_output_folder
            )
        else:
            self._plotter = None

        # check if params are passed with/converted to consistent length
        # (== self.n_levels)
        if not are_of_same_length(
            self.scale_factors,
            self.iterations,
            self.tau,
            self.regularization_sigma,
        ):
            raise ValueError("Inconsistent lengths of passed parameters")

        self._metrics = []
        self._counter = 0

        self._image_shape = None
        self._full_size_spatial_transformer = None

        if force_type == "demons":
            self._forces_layer = DemonForces(method=self.forces)
        else:
            raise NotImplementedError(
                f"Registration variant {force_type} is not implemented"
            )

    @property
    def n_levels(self):
        return len(self.scale_factors)

    @property
    def config(self):
        return dict(
            iterations=self.iterations,
            tau=self.tau,
            regularization_sigma=self.regularization_sigma,
        )

    def _update_step(
        self,
        level: int,
        iteration: int,
        scale_factor: float,
        vector_field: torch.Tensor,
        moving_image: torch.Tensor,
        warped_image: torch.Torch,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        forces = self._forces_layer(
            warped_image,
            fixed_image,
            moving_mask,
            fixed_mask,
            original_image_spacing,
            use_masks=self.use_masks[level],
        )
        decayed_tau = exponential_decay(
            initial_value=self.tau[level],
            i_level=level,
            i_iteration=iteration,
            level_lambda=self.tau_level_decay,
            iteration_lambda=self.tau_iteration_decay,
        )

        vector_field = vector_field + decayed_tau * forces

        decayed_sigma = tuple(
            exponential_decay(
                initial_value=s,
                i_level=level,
                i_iteration=iteration,
                level_lambda=self.sigma_level_decay,
                iteration_lambda=self.sigma_iteration_decay,
            )
            for s in self.regularization_sigma[level]
        )
        n_spatial_dims = vector_field.shape[1]
        sigma_cutoff = (2.0, 2.0, 2.0)[:n_spatial_dims]
        gaussian_smoothing = self._GAUSSIAN_SMOOTHING[n_spatial_dims]
        _regularization_layer = gaussian_smoothing(
            sigma=decayed_sigma,
            sigma_cutoff=sigma_cutoff,
            force_same_size=True,
            spacing=self.original_image_spacing,
            use_image_spacing=self.use_image_spacing,
        ).to(vector_field)

        vector_field = _regularization_layer(vector_field)

        return vector_field

    def _run_registration(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        moving_landmarks: torch.Tensor | None = None,
        fixed_landmarks: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
        yield_each_step: bool = False,
    ) -> Generator[dict[str]]:
        device = moving_image.device
        if moving_image.ndim != fixed_image.ndim:
            raise RuntimeError("Dimension mismatch betwen moving and fixed image")
        # define dimensionalities and shapes
        n_image_dimensions = moving_image.ndim
        n_spatial_dims = n_image_dimensions - 2  # -1 batch dim, -1 color dim

        full_uncropped_shape = tuple(fixed_image.shape)

        original_moving_image = moving_image
        original_moving_mask = moving_mask

        has_initial_vector_field = initial_vector_field is not None
        original_initial_vector_field = initial_vector_field

        if self.restrict_to_mask_bbox and (
            moving_mask is not None or fixed_mask is not None
        ):
            masks = [m for m in (moving_mask, fixed_mask) if m is not None]
            if len(masks) == 2:
                # we compute the union of both masks to get the overall bounding box
                union_mask = torch.logical_or(*masks)
            else:
                union_mask = masks[0]

            bbox = get_bounding_box(union_mask, padding=5)
            self.logger.debug(f"Restricting registration to bounding box {bbox}")

            moving_image = moving_image[bbox]
            fixed_image = fixed_image[bbox]
            if moving_mask is not None:
                moving_mask = moving_mask[bbox]
            if fixed_mask is not None:
                fixed_mask = fixed_mask[bbox]
            if has_initial_vector_field:
                initial_vector_field = initial_vector_field[(..., *bbox[2:])]
        else:
            bbox = ...

        if moving_mask is not None:
            moving_mask = torch.as_tensor(moving_mask, dtype=torch.bool)
        if fixed_mask is not None:
            fixed_mask = torch.as_tensor(fixed_mask, dtype=torch.bool)

        # create new spatial transformers if needed (skip batch and color dimension)
        self._create_spatial_transformers(
            fixed_image.shape[2:], device=fixed_image.device
        )

        full_size_moving = moving_image
        full_cropped_shape = tuple(fixed_image.shape)
        full_cropped_spatial_shape = full_cropped_shape[2:]

        # for tracking level-wise metrics (used for early stopping, if enabled)
        metrics = []
        vector_field = None
        warped_fixed_landmarks = None

        metric_before = self._calculate_metric(
            moving_image=moving_image, fixed_image=fixed_image, fixed_mask=fixed_mask
        )

        for i_level, (scale_factor, iterations) in enumerate(
            zip(self.scale_factors, self.iterations)
        ):
            self.logger.info(
                f"Start level {i_level + 1}/{self.n_levels} with {scale_factor=} and {iterations=}"
            )
            self._counter = 0

            (
                scaled_moving_image,
                scaled_fixed_image,
                scaled_moving_mask,
                scaled_fixed_mask,
            ) = self._perform_scaling(
                moving_image,
                fixed_image,
                moving_mask,
                fixed_mask,
                scale_factor=scale_factor,
            )

            if vector_field is None:
                # create an empty (all zero) vector field
                vector_field = torch.zeros(
                    scaled_fixed_image.shape[:1]
                    + (n_spatial_dims,)
                    + scaled_fixed_image.shape[2:],
                    device=moving_image.device,
                )
            else:
                # this will also scale the initial (full scale) vector field in the
                # first iteration
                vector_field = match_vector_field(vector_field, scaled_fixed_image)

            if has_initial_vector_field:
                scaled_initial_vector_field = match_vector_field(
                    initial_vector_field, scaled_fixed_image
                )
            else:
                scaled_initial_vector_field = None

            spatial_transformer = self.get_submodule(
                f"spatial_transformer_level_{i_level}",
            )

            level_metrics = []

            for i_iteration in range(iterations):
                t_step_start = time.time()
                if has_initial_vector_field:
                    # if we have an initial vector field: compose both vector fields.
                    # Here: resolution defined by given registration level
                    composed_vector_field = vector_field + spatial_transformer(
                        scaled_initial_vector_field, vector_field
                    )
                else:
                    composed_vector_field = vector_field

                warped_moving = spatial_transformer(
                    scaled_moving_image,
                    composed_vector_field,
                )
                warped_scaled_moving_mask = spatial_transformer(
                    scaled_moving_mask, composed_vector_field
                )
                if self.debug:
                    cropped_fixed_landmarks = fixed_landmarks - [
                        bbox[i].start for i in range(2, n_image_dimensions)
                    ]
                    exact_scale_factor = match_vector_field(
                        vector_field=composed_vector_field,
                        image=fixed_image,
                        return_scale_factor=True,
                    )
                    scaled_fixed_landmarks = torch.as_tensor(
                        cropped_fixed_landmarks / exact_scale_factor
                    ).to(device)
                    warped_fixed_landmarks = TRELoss._warped_fixed_landmarks(
                        fixed_landmarks=scaled_fixed_landmarks,
                        vector_field=composed_vector_field,
                    )
                    warped_fixed_landmarks = (
                        warped_fixed_landmarks.detach().cpu() * exact_scale_factor
                    ).numpy()
                    warped_fixed_landmarks = warped_fixed_landmarks + [
                        bbox[i].start for i in range(2, n_image_dimensions)
                    ]

                level_metrics.append(
                    self._calculate_metric(
                        moving_image=warped_moving,
                        fixed_image=scaled_fixed_image,
                        fixed_mask=scaled_fixed_mask,
                    )
                )

                # one vector field update step
                if yield_each_step:
                    vector_field_before = vector_field.clone()

                vector_field = self._update_step(
                    level=i_level,
                    iteration=i_iteration,
                    scale_factor=scale_factor,
                    vector_field=vector_field,
                    moving_image=scaled_moving_image,
                    warped_image=warped_moving,
                    fixed_image=scaled_fixed_image,
                    moving_mask=warped_scaled_moving_mask,
                    fixed_mask=scaled_fixed_mask,
                    original_image_spacing=original_image_spacing,
                    initial_vector_field=scaled_initial_vector_field,
                )

                if yield_each_step:
                    yield dict(
                        type="intermediate",
                        level=i_level,
                        iteration=i_iteration,
                        moving_image=scaled_moving_image,
                        fixed_image=scaled_fixed_image,
                        warped_image=warped_moving,
                        moving_mask=scaled_moving_mask,
                        fixed_mask=scaled_fixed_mask,
                        warped_mask=warped_scaled_moving_mask,
                        vector_field_before=vector_field_before,
                        vector_field_after=vector_field,
                    )

                # check early stopping
                if self._check_early_stopping(metrics=level_metrics, i_level=i_level):
                    break

                log = {
                    "level": i_level,
                    "iteration": i_iteration,
                    # "tau": decayed_tau,
                    "metric": level_metrics[-1],
                }

                t_step_end = time.time()
                log["step_runtime"] = t_step_end - t_step_start
                self.logger.debug(log)

                # here we do the debug stuff
                if self.debug and (
                    i_iteration % self.debug_step_size == 0 or i_iteration < 10
                ):
                    debug_metrics = {
                        "metric": level_metrics[-1],
                        "level_image_shape": tuple(scaled_moving_image.shape),
                        "vector_field": {},
                    }

                    dim_names = ("x", "y", "z")
                    for i_dim in range(n_spatial_dims):
                        debug_metrics["vector_field"][dim_names[i_dim]] = {
                            "min": float(torch.min(vector_field[:, i_dim])),
                            # "Q0.05": float(
                            #     torch.quantile(vector_field[:, i_dim], 0.05)
                            # ),
                            "mean": float(torch.mean(vector_field[:, i_dim])),
                            # "Q0.95": float(
                            #     torch.quantile(vector_field[:, i_dim], 0.95)
                            # ),
                            "max": float(torch.max(vector_field[:, i_dim])),
                        }
                    self._plotter.save_snapshot(
                        moving_image=scaled_moving_image,
                        fixed_image=scaled_fixed_image,
                        warped_image=warped_moving,
                        forces=None,
                        vector_field=vector_field,
                        moving_mask=scaled_moving_mask,
                        fixed_mask=scaled_fixed_mask,
                        warped_mask=warped_scaled_moving_mask,
                        full_spatial_shape=full_cropped_spatial_shape,
                        stage="vroc",
                        level=i_level,
                        scale_factor=scale_factor,
                        iteration=i_iteration,
                        metrics=debug_metrics,
                        output_folder=self.debug_output_folder,
                    )
                    write_landmarks(
                        landmarks=fixed_landmarks,
                        filepath=self.debug_output_folder
                        / f"warped_landmarks_initial.csv",
                    )
                    write_landmarks(
                        landmarks=warped_fixed_landmarks,
                        filepath=self.debug_output_folder
                        / f"warped_landmarks_level_{i_level:02d}_iteration_{i_iteration:04d}.csv",
                    )

                    self.logger.debug(
                        f"Created snapshot of registration at level={i_level}, iteration={i_iteration}"
                    )

            metrics.append(level_metrics)

        vector_field = match_vector_field(vector_field, full_size_moving)

        if self.restrict_to_mask_bbox:
            # undo restriction to mask, i.e. insert results into full size data
            _vector_field = torch.zeros(
                vector_field.shape[:2] + original_moving_image.shape[2:],
                device=vector_field.device,
            )
            _vector_field[(...,) + bbox[2:]] = vector_field
            vector_field = _vector_field

        spatial_transformer = SpatialTransformer(
            shape=full_uncropped_shape[2:], default_value=self.default_voxel_value
        ).to(fixed_image.device)

        result = {"type": "final"}

        if initial_vector_field is not None:
            # if we have an initial vector field: compose both vector fields.
            # Here: at full resolution without cropping/bbox
            composed_vector_field = vector_field + spatial_transformer(
                original_initial_vector_field, vector_field
            )
            result["vector_fields"] = [original_initial_vector_field, vector_field]
        else:
            composed_vector_field = vector_field
            result["vector_fields"] = [vector_field]

        result["composed_vector_field"] = composed_vector_field

        warped_moving_image = spatial_transformer(
            original_moving_image, composed_vector_field
        )
        if original_moving_mask is not None:
            warped_moving_mask = spatial_transformer(
                original_moving_mask, composed_vector_field
            )
        else:
            warped_moving_mask = None

        if initial_vector_field is not None:
            warped_affine_moving_image = spatial_transformer(
                original_moving_image, original_initial_vector_field
            )

            if original_moving_mask is not None:
                warped_affine_moving_mask = spatial_transformer(
                    original_moving_mask, original_initial_vector_field
                )
            else:
                warped_affine_moving_mask = None
        else:
            warped_affine_moving_image = None
            warped_affine_moving_mask = None

        metric_after = self._calculate_metric(
            moving_image=warped_moving_image[bbox],
            fixed_image=fixed_image,
            fixed_mask=fixed_mask,
        )

        if self.debug:
            debug_info = {
                "metric_before": metric_before,
                "metric_after": metric_after,
                "level_metrics": metrics,
            }

        else:
            debug_info = None
        result["debug_info"] = debug_info
        result["warped_moving_image"] = warped_moving_image
        result["warped_moving_mask"] = warped_moving_mask
        result["warped_affine_moving_mask"] = warped_affine_moving_mask
        result["warped_affine_moving_image"] = warped_affine_moving_image

        yield result

    def run_registration(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        moving_landmarks: torch.Tensor | None = None,
        fixed_landmarks: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
        yield_each_step: bool = False,
    ):
        # registration_generator is either a generator yielding each registration step
        # or just a generator of length 1 yielding the final result
        registration_generator = self._run_registration(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            moving_landmarks=moving_landmarks,
            fixed_landmarks=fixed_landmarks,
            original_image_spacing=original_image_spacing,
            initial_vector_field=initial_vector_field,
            yield_each_step=yield_each_step,
        )

        if yield_each_step:

            def yield_registration_steps():
                yield from registration_generator

            return yield_registration_steps()
        else:
            return next(registration_generator)

    @timing()
    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        original_image_spacing: FloatTuple3D,
    ):
        return self.run_registration(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            original_image_spacing=original_image_spacing,
        )
