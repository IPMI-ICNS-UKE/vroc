from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn

from vroc.helper import batch_array, rescale_range


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


if __name__ == "__main__":
    from vroc.models import UNet

    lung_segmenter = LungSegmenter2D(
        model=UNet().to("cuda"),
        state_filepath=Path(
            "/home/tsentker/Documents/projects/ebolagnul/lung_segmenter.pth"
        ),
    )
