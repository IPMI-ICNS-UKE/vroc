from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Literal, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

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
from vroc.guesser import ParameterGuesser
from vroc.helper import to_one_hot
from vroc.interpolation import rescale, resize
from vroc.logger import LoggerMixin, RegistrationLogEntry
from vroc.loss import DiceLoss, TRELoss, WarpedMSELoss, smooth_vector_field_loss
from vroc.models import VarReg


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
        use_masks: MaybeSequence[bool] = True,
        image_spacing: FloatTuple2D | FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: ArrayOrTensor | None = None,
        register_affine: bool = True,
        affine_loss_function: Callable | None = None,
        affine_iterations: int = 300,
        affine_step_size: float = 1e-3,
        force_type: Literal["demons", "ncc", "ngf"] = "demons",
        gradient_type: Literal["active", "passive", "dual"] = "dual",
        segment_roi: bool = False,
        valid_value_range: Tuple[Number, Number] | None = None,
        early_stopping_delta: float = 0.0,
        early_stopping_window: int | None = None,
        default_parameters: dict | None = None,
        return_as_tensor: bool = False,
        debug: bool = False,
        debug_output_folder: PathLike | None = None,
        debug_step_size: int = 10,
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
            )

        # handle ROIs
        # passed masks overwrite ROI segmenter
        # check if roi_segmenter is given if segment_roi
        if segment_roi and not self.roi_segmenter:
            raise RuntimeError("Please pass a ROI segmenter")

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

        varreg = VarReg(
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
        ).to(self.device)

        with torch.autocast(device_type="cuda", enabled=True), torch.inference_mode():
            varreg_result = varreg.run_registration(
                moving_image=moving_image,
                fixed_image=fixed_image,
                moving_mask=moving_mask,
                fixed_mask=fixed_mask,
                original_image_spacing=image_spacing,
                initial_vector_field=initial_vector_field,
            )

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

    @timing()
    def register_and_train_boosting(
        self,
        # boosting specific args/kwargs
        model: nn.Module,
        optimizer: optim.Optimizer,
        n_iterations: int,
        moving_image: ArrayOrTensor,
        fixed_image: ArrayOrTensor,
        boost_scale: float = 1.0,
        moving_keypoints: torch.Tensor | None = None,
        fixed_keypoints: torch.Tensor | None = None,
        moving_labels: torch.Tensor | None = None,
        fixed_labels: torch.Tensor | None = None,
        n_label_classes: int | None = None,
        image_loss_function: Literal["mse"] | None = None,
        keypoint_loss_weight: float = 1.0,
        label_loss_weight: float = 1.0,
        image_loss_weight: float = 1.0,
        smoothness_loss_weight: float = 1.0,
        # registration kwargs as in register(...)
        moving_mask: ArrayOrTensor | None = None,
        fixed_mask: ArrayOrTensor | None = None,
        use_masks: MaybeSequence[bool] = True,
        image_spacing: FloatTuple2D | FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: ArrayOrTensor | None = None,
        register_affine: bool = True,
        affine_loss_function: Callable | None = None,
        affine_iterations: int = 300,
        affine_step_size: float = 1e-3,
        force_type: Literal["demons", "ncc", "ngf"] = "demons",
        gradient_type: Literal["active", "passive", "dual"] = "dual",
        segment_roi: bool = False,
        valid_value_range: Tuple[Number, Number] | None = None,
        early_stopping_delta: float = 0.0,
        early_stopping_window: int | None = None,
        default_parameters: dict | None = None,
        return_as_tensor: bool = False,
        debug: bool = False,
        debug_output_folder: PathLike | None = None,
        debug_step_size: int = 10,
    ) -> RegistrationResult:
        # TODO: duplicate code fragment
        # TODO: preprocessing, checks and as_tensor as separate function
        # n_spatial_dims is defined by length of image_spacing
        n_spatial_dims = len(image_spacing)
        # n_total_dims = n_spatial_dims + batch dim + color dim
        n_total_dims = n_spatial_dims + 2

        # check if any optimization target is passed (i.e., either keypoints or labels)
        possible_targets = (
            moving_keypoints,
            fixed_keypoints,
            moving_labels,
            fixed_labels,
        )
        if not any(target is not None for target in possible_targets):
            raise ValueError(
                "Please pass at least one moving and fixed "
                "target (keypoints or labels)"
            )

        registration_result = self.register(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            use_masks=use_masks,
            image_spacing=image_spacing,
            initial_vector_field=initial_vector_field,
            register_affine=register_affine,
            affine_loss_function=affine_loss_function,
            affine_iterations=affine_iterations,
            affine_step_size=affine_step_size,
            force_type=force_type,
            gradient_type=gradient_type,
            segment_roi=segment_roi,
            valid_value_range=valid_value_range,
            early_stopping_delta=early_stopping_delta,
            early_stopping_window=early_stopping_window,
            default_parameters=default_parameters,
            return_as_tensor=True,  # we need tensors for boosting
            debug=debug,
            debug_output_folder=debug_output_folder,
            debug_step_size=debug_step_size,
        )

        model = model.to(self.device)

        moving_image = torch.clone(registration_result.moving_image)
        fixed_image = torch.clone(registration_result.fixed_image)

        # transform and clip images to value range [0, 1]
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

        # init loss layers and values (even if not used)

        tre_loss = TRELoss(apply_sqrt=False, reduction=None)
        dice_loss = DiceLoss(include_background=True, reduction="mean")

        # setup and initialize losses and loss weights
        requested_losses = set()
        loss_weights = {}
        if (
            keypoint_loss_weight != 0
            and moving_keypoints is not None
            and fixed_keypoints is not None
        ):
            # we have to compute keypoint loss
            requested_losses.add("keypoint")
            loss_weights["keypoint"] = keypoint_loss_weight
            # calculate loss before boosting
            tre_loss_before_boosting = tre_loss(
                composed_vector_field,
                moving_keypoints,
                fixed_keypoints,
                image_spacing,
            )
            tre_metric_before_boosting = tre_loss_before_boosting.sqrt().mean()

        if smoothness_loss_weight != 0:
            # we have to compute smoothness loss
            requested_losses.add("smoothness")
            loss_weights["smoothness"] = smoothness_loss_weight
            # calculate loss before boosting
            smoothness_before_boosting = smooth_vector_field_loss(
                vector_field=composed_vector_field, mask=fixed_mask, l2r_variant=True
            )
        if image_loss_weight != 0 and image_loss_function:
            # we have to compute image loss
            if image_loss_function == "mse":
                image_loss = WarpedMSELoss()
            else:
                raise NotImplementedError

            requested_losses.add("image")
            loss_weights["image"] = image_loss_weight
            # calculate loss before boosting
            image_loss_before_boosting = image_loss(
                moving_image=moving_image,
                vector_field=composed_vector_field,
                fixed_image=fixed_image,
                fixed_mask=fixed_mask,
            )
        if (
            label_loss_weight != 0
            and moving_labels is not None
            and fixed_labels is not None
        ):
            # convert to tensors
            moving_labels = as_tensor(
                moving_labels, n_dim=n_total_dims, dtype=torch.uint8, device=self.device
            )
            fixed_labels = as_tensor(
                fixed_labels, n_dim=n_total_dims, dtype=torch.uint8, device=self.device
            )

            # convert to one hot if not already
            # We use float32 here, as there are no gradients with NN interpolation.
            # This must be a bug in PyTorch
            if moving_labels.shape[1] == 1:
                moving_labels = to_one_hot(
                    moving_labels, n_classes=n_label_classes, dtype=torch.float32
                )
            if fixed_labels.shape[1] == 1:
                fixed_labels = to_one_hot(
                    fixed_labels, n_classes=n_label_classes, dtype=torch.float32
                )

            # we have to compute label loss
            requested_losses.add("label")
            loss_weights["label"] = label_loss_weight
            # calculate loss before boosting
            warped_labels = model.spatial_transformer(
                image=moving_labels,
                transformation=composed_vector_field,
                default_value=0,
                mode="nearest",
            )
            dice_loss_before_boosting = dice_loss(warped_labels, fixed_labels)

        gradient_scaler = torch.cuda.amp.GradScaler()
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            threshold=1e-4,
            threshold_mode="rel",
            factor=0.1,
            patience=10,
            cooldown=10,
            min_lr=1e-5,
        )

        if debug:
            from vroc.plot import RegistrationProgressPlotter

            debug_output_folder = Path(debug_output_folder)
            debug_output_folder.mkdir(exist_ok=True, parents=True)
            plotter = RegistrationProgressPlotter(output_folder=debug_output_folder)
        else:
            plotter = None

        max_iteration_length = len(str(n_iterations))

        # initial values in case of 0 iterations
        composed_boosted_vector_field = composed_vector_field
        vector_field_boost = torch.zeros_like(composed_boosted_vector_field)

        for i_iteration in range(n_iterations):
            # reset after 20 iterations
            # if i_iteration % 20 == 0:
            #     print(f'reset at {i_iteration = }')
            #     composed_boosted_vector_field = composed_vector_field

            # is_train = i_iteration < n_iterations
            is_train = True
            maybe_enabled_grad = nullcontext() if is_train else torch.inference_mode()

            with maybe_enabled_grad:
                optimizer.zero_grad()
                # composed_boosted_vector_field = composed_boosted_vector_field.detach()
                with torch.autocast(device_type="cuda", enabled=True):

                    original_shape = moving_image.shape[2:]
                    boosting_shape = tuple(
                        int(round(s * boost_scale)) for s in moving_image.shape[2:]
                    )

                    vector_field_boost = model(
                        resize(moving_image, output_shape=boosting_shape, order=1),
                        resize(fixed_image, output_shape=boosting_shape, order=1),
                        resize(moving_mask, output_shape=boosting_shape, order=0),
                        resize(fixed_mask, output_shape=boosting_shape, order=0),
                        resize(
                            composed_vector_field, output_shape=boosting_shape, order=1
                        )
                        * boost_scale,
                        image_spacing,
                        n_iterations=None if is_train else 100,
                    )
                    vector_field_boost = resize(
                        vector_field_boost / boost_scale,
                        output_shape=original_shape,
                        order=1,
                    )

                    # start from scratch
                    composed_boosted_vector_field = (
                        vector_field_boost
                        + model.spatial_transformer(
                            composed_vector_field, vector_field_boost
                        )
                    )

                    # continue with boosted result

                    # composed_boosted_vector_field = (
                    #     vector_field_boost
                    #     + model.spatial_transformer(
                    #         composed_boosted_vector_field, vector_field_boost
                    #     )
                    # )

                    # initialize log entry
                    log = RegistrationLogEntry(
                        stage="boosting",
                        iteration=i_iteration,
                    )

                    # compute specified losses (keypoints, smoothness and labels)
                    losses = {}
                    if "keypoint" in requested_losses:
                        tre_loss_after_boosting = tre_loss(
                            composed_boosted_vector_field,
                            moving_keypoints,
                            fixed_keypoints,
                            image_spacing,
                        )

                        tre_metric_after_boosting = (
                            tre_loss_after_boosting.sqrt().mean()
                        )
                        tre_ratio_loss = (
                            tre_metric_after_boosting / tre_metric_before_boosting
                        )

                        losses["keypoint"] = tre_ratio_loss

                        # add to log
                        log.tre_metric_before_boosting = tre_metric_before_boosting
                        log.tre_metric_after_boosting = tre_metric_after_boosting

                    if "smoothness" in requested_losses:
                        smoothness_after_boosting = smooth_vector_field_loss(
                            vector_field=composed_boosted_vector_field,
                            mask=fixed_mask,
                            l2r_variant=True,
                        )

                        smoothness_ratio_loss = (
                            smoothness_after_boosting / smoothness_before_boosting
                        )

                        losses["smoothness"] = (
                            torch.relu(smoothness_ratio_loss - 1) + 1.0
                        )

                        # add to log
                        log.smoothness_before_boosting = smoothness_before_boosting
                        log.smoothness_after_boosting = smoothness_after_boosting

                        # ratio < 1: better, ratio > 1 worse
                        # only penalize worsening of smoothness
                        # if smoothness_ratio_loss < 1:
                        #     smoothness_ratio_loss = 0.5 * smoothness_ratio_loss + 0.5
                        # smoothness_ratio_loss = torch.maximum(
                        #     smoothness_ratio_loss, torch.as_tensor(1.0)
                        # )

                    if "image" in requested_losses:
                        image_loss_after_boosting = image_loss(
                            moving_image=moving_image,
                            vector_field=composed_boosted_vector_field,
                            fixed_image=fixed_image,
                            fixed_mask=fixed_mask,
                        )

                        image_ratio_loss = (
                            image_loss_after_boosting / image_loss_before_boosting
                        )

                        losses["image"] = image_ratio_loss

                        # add to log
                        log.image_loss_before_boosting = image_loss_before_boosting
                        log.image_loss_after_boosting = image_loss_after_boosting

                    if "label" in requested_losses:
                        warped_labels = model.spatial_transformer(
                            moving_labels,
                            composed_boosted_vector_field,
                            default_value=0,
                            mode="bilinear",
                        )

                        dice_loss_after_boosting = dice_loss(
                            warped_labels, fixed_labels
                        )

                        dice_ratio_loss = (
                            dice_loss_after_boosting / dice_loss_before_boosting
                        )
                        losses["label"] = dice_ratio_loss

                        # add to log
                        log.dice_loss_before_boosting = dice_loss_before_boosting
                        log.dice_loss_after_boosting = dice_loss_after_boosting

                    # reduce losses to scalar
                    loss = 0.0
                    weight_sum = 0.0
                    for loss_name, loss_value in losses.items():
                        loss += loss_weights[loss_name] * loss_value
                        weight_sum += loss_weights[loss_name]
                    loss /= weight_sum

                    # add loss info to log
                    log.loss = loss
                    log.losses = losses
                    log.loss_weights = loss_weights
                    log.learning_rate = optimizer.param_groups[0]["lr"]
                    self.logger.debug(log)

            if is_train:
                gradient_scaler.scale(loss).backward()
                gradient_scaler.step(optimizer)
                gradient_scaler.update()

                scheduler.step(loss)

            # here we do the debug stuff
            if debug and (
                i_iteration % debug_step_size == 0 or i_iteration == n_iterations - 1
            ):
                spatial_shape = moving_image.shape[2:]
                n_spatial_dims = len(spatial_shape)

                with torch.inference_mode():
                    # warp moving image and mask
                    warped_moving_image = model.spatial_transformer(
                        moving_image, composed_boosted_vector_field
                    )
                    warped_moving_mask = model.spatial_transformer(
                        moving_mask, composed_boosted_vector_field
                    )

                debug_metrics = {"vector_field": {}}

                dim_names = ("x", "y", "z")
                for i_dim in range(n_spatial_dims):
                    debug_metrics["vector_field"][dim_names[i_dim]] = {
                        "min": float(
                            torch.min(composed_boosted_vector_field[:, i_dim])
                        ),
                        # "Q0.05": float(
                        #     torch.quantile(
                        #         boosted_vector_field_without_affine[:, i_dim], 0.05
                        #     )
                        # ),
                        "mean": float(
                            torch.mean(composed_boosted_vector_field[:, i_dim])
                        ),
                        # "Q0.95": float(
                        #     torch.quantile(
                        #         boosted_vector_field_without_affine[:, i_dim], 0.95
                        #     )
                        # ),
                        "max": float(
                            torch.max(composed_boosted_vector_field[:, i_dim])
                        ),
                    }
                i_level = 0
                stage = "boosting"
                plotter.save_snapshot(
                    moving_image=moving_image,
                    fixed_image=fixed_image,
                    warped_image=warped_moving_image,
                    forces=None,
                    vector_field=composed_boosted_vector_field,
                    moving_mask=moving_mask,
                    fixed_mask=fixed_mask,
                    warped_mask=warped_moving_mask,
                    full_spatial_shape=spatial_shape,
                    stage="boosting",
                    level=i_level,
                    scale_factor=1.0,
                    iteration=i_iteration,
                    metrics=debug_metrics,
                    output_folder=debug_output_folder,
                )
                self.logger.debug(
                    f"Created snapshot of registration stage={stage} at iteration={i_iteration}"
                )

        registration_result.composed_vector_field = composed_boosted_vector_field
        # warp moving image with composed boosted vector field
        warped_moving_image = model.spatial_transformer(
            moving_image, composed_boosted_vector_field
        )
        warped_moving_image = (
            warped_moving_image * (valid_value_range[1] - valid_value_range[0])
        ) + valid_value_range[0]
        registration_result.warped_moving_image = warped_moving_image
        registration_result.vector_fields.append(vector_field_boost)

        if not return_as_tensor:
            registration_result.to_numpy()

        return registration_result
