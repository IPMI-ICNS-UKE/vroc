from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn

from vroc.common_types import TorchDevice
from vroc.helper import batch_array, nearest_factor_pow_2, rescale_range
from vroc.interpolation import resize
from vroc.preprocessing import crop_or_pad


class Segmenter2D(ABC):
    def __init__(
        self,
        model: nn.Module,
        state_filepath: Path = None,
        device="cuda",
        iter_axis: int = 2,  # Model input: axial slices
    ):
        self.model = model
        self.device = device
        self.iter_axis = iter_axis

        if state_filepath:
            try:
                self._load_state_dict(state_filepath)
            except RuntimeError:
                self._load_state_dict(state_filepath, remove_module=True)

        self.model.to(self.device)
        self.model.eval()

    def _load_state_dict(self, state_filepath, remove_module: bool = True):
        state_dict = torch.load(state_filepath, map_location=self.device)
        if remove_module:
            _state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("module."):
                    _state_dict[key[7:]] = value
            state_dict = _state_dict
        self.model.load_state_dict(state_dict)

    def _prepare_axes(self, image: np.ndarray, inverse: bool = False):
        if inverse:
            image = np.swapaxes(image, 0, self.iter_axis)
            image = np.flip(image, axis=1)
        else:
            image = np.flip(image, axis=1)
            image = np.swapaxes(image, self.iter_axis, 0)
        return image

    def segment(
        self,
        image: np.ndarray,
        batch_size: int = 16,
        fill_holes: bool = True,
        clear_cuda_cache: bool = False,
    ):
        image = self._prepare_axes(image=image, inverse=False)
        image = self._prepare_image(image=image)
        predicted_batches = []
        for image_batch in batch_array(image, batch_size=batch_size):
            image_batch = torch.as_tensor(
                image_batch, dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                logits = self.model(image_batch)
            prediction = torch.sigmoid(logits) > 0.3
            prediction = prediction.detach().cpu().numpy()
            prediction = prediction.squeeze(axis=1)
            predicted_batches.append(prediction)

        prediction = np.concatenate(predicted_batches)
        prediction = self._prepare_axes(image=prediction, inverse=True)
        if fill_holes:
            prediction = ndi.binary_fill_holes(prediction)

        if clear_cuda_cache:
            torch.cuda.empty_cache()

        return prediction

    @abstractmethod
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        pass


class LungSegmenter2D(Segmenter2D):
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = rescale_range(image, input_range=(-1000, 200), output_range=(-1, 1))
        return image[:, np.newaxis]


class LungSegmenter3D:
    def __init__(self, model: nn.Module, device: TorchDevice):
        self.model = model.to(device)
        self.device = device

    def segment(self, image: np.ndarray, subsample: float = 2.0) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 3:
            raise ValueError("Please pass a 3D image")

        original_shape = image.shape

        image = resize(
            image, output_shape=tuple(s // subsample for s in original_shape)
        )
        unpadded_shape = image.shape
        padded_shape = tuple(nearest_factor_pow_2(s) for s in unpadded_shape)

        image, _ = crop_or_pad(image=image, mask=None, target_shape=padded_shape)
        image = rescale_range(
            image,
            input_range=(-1024, 3071),
            output_range=(0, 1),
            clip=True,
        )
        image = torch.as_tensor(image[None, None], device=self.device)

        self.model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = self.model(image)
            prediction = torch.sigmoid(prediction)

        prediction = prediction.detach().cpu().numpy().squeeze(axis=(0, 1))
        prediction, _ = crop_or_pad(
            image=prediction, mask=None, target_shape=unpadded_shape
        )

        prediction = image = resize(prediction, output_shape=original_shape)
        prediction = prediction > 0.5

        return prediction
