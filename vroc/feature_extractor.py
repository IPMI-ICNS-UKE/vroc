import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from vroc.common_types import FloatTuple3D
from vroc.interpolation import resize_spacing
from vroc.models import AutoEncoder, VarReg3d
from vroc.oriented_histogram import OrientedHistogram


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class OrientedHistogramFeatureExtrator(FeatureExtractor):
    def __init__(self, n_bins: int = 16, device: str = "cuda"):
        self.n_bins = n_bins
        self.device = device

    @staticmethod
    def calculate_oriented_histogram(
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        fixed_mask: np.ndarray,
        moving_mask: np.ndarray,
        image_spacing: FloatTuple3D,
        target_image_spacing: FloatTuple3D = (4.0, 4.0, 4.0),
        n_bins: int = 16,
        device: str = "cuda",
    ):
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
            fixed_mask.astype(np.uint8),
            input_image_spacing=image_spacing,
            output_image_spacing=target_image_spacing,
            order=0,
        )
        moving_mask = resize_spacing(
            moving_mask.astype(np.uint8),
            input_image_spacing=image_spacing,
            output_image_spacing=target_image_spacing,
            order=0,
        )

        fixed_mask = np.asarray(fixed_mask, dtype=bool)
        moving_mask = np.asarray(moving_mask, dtype=bool)

        union_mask = fixed_mask | moving_mask

        _fixed_image = torch.as_tensor(fixed_image, device=device)[None, None]
        _moving_image = torch.as_tensor(moving_image, device=device)[None, None]

        _fixed_mask = torch.as_tensor(fixed_mask, device=device)[None, None]
        _moving_mask = torch.as_tensor(moving_mask, device=device)[None, None]
        _union_mask = torch.as_tensor(union_mask, device=device)[None, None]
        _mask = torch.ones_like(_fixed_image)

        varreg = VarReg3d(
            iterations=200,
            scale_factors=1.0,
            demon_forces="symmetric",
            tau=1.0,
            regularization_sigma=(1, 1, 1),
            restrict_to_mask_bbox=False,
        ).to(device)

        with torch.no_grad():
            warped_moving_image, vector_field, misc = varreg.forward(
                _fixed_image, _mask, _moving_image, (1.0, 1.0, 1.0)
            )
        vector_field1 = vector_field.detach().cpu().numpy().squeeze(axis=0)
        oh1 = OrientedHistogram(n_bins=n_bins).calculate(vector_field1, mask=union_mask)

        with torch.no_grad():
            warped_moving_image, vector_field, misc = varreg.forward(
                _moving_image, _mask, _fixed_image, (1.0, 1.0, 1.0)
            )
        vector_field2 = vector_field.detach().cpu().numpy().squeeze(axis=0)

        oh2 = OrientedHistogram(n_bins=n_bins).calculate(vector_field2, mask=union_mask)

        return np.mean(
            (
                oh1,
                oh2,
            ),
            axis=0,
        )

    def extract(
        self,
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        fixed_mask: np.ndarray,
        moving_mask: np.ndarray,
        image_spacing: FloatTuple3D,
        target_image_spacing: FloatTuple3D = (4.0, 4.0, 4.0),
    ) -> np.ndarray:
        features = OrientedHistogramFeatureExtrator.calculate_oriented_histogram(
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


# class FeatureExtractor:
#     def __init__(self, state_filepath):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.model = torch.load(model_path, map_location=torch.device(self.device))
#         self.model = AutoEncoder().to(device=self.device)
#         self.model.load_state_dict(
#             torch.load(state_filepath, map_location=torch.device(self.device))[
#                 "state_dict"
#             ]
#         )
#         self.model.eval()
#
#     def infer(self, image: np.ndarray):
#         image = self._prepare_image(image=image)
#         image = self._prepare_torch_tensor(image)
#         with torch.no_grad():
#             _, embedded = self.model(image)
#         return embedded
#
#     def _prepare_image(self, image: np.ndarray) -> np.ndarray:
#         return rescale_range(image, input_range=(-1024, 3071), output_range=(0, 1))
#
#     def _prepare_torch_tensor(self, image: np.ndarray) -> torch.tensor:
#         image = torch.as_tensor(image.copy(), dtype=torch.float32, device=self.device)
#         return image[None, None, :]


if __name__ == "__main__":
    feature_extractor = FeatureExtractor(
        state_filepath="/home/tsentker/Documents/results/vroc_AE/models/epoch075_val_loss_=_0.004.pth"
    )
    root_dir = (
        Path("/home/tsentker/data"),
        Path("/datalake/NLST"),
    )
    root_dir = next(p for p in root_dir if p.exists())

    dataset_paths = [
        "dirlab2022/data/**/Images",
    ]

    dataset_paths = [os.path.join(root_dir, path) for path in dataset_paths]
    filepaths = AutoencoderDataset.fetch_filepaths(dataset_paths)

    features = {}
    for filepath in tqdm(filepaths):
        img = AutoencoderDataset.load_and_preprocess(filepath=filepath)
        feature_vector = feature_extractor.infer(image=img)
        case = re.search(r"Case(\d*)Pack", str(filepath))[1]
        features[(case, os.path.splitext(os.path.basename(filepath))[0])] = (
            feature_vector.detach().cpu().numpy()
        )

# def extract_histogram_features(image: np.ndarray):
#     lower_percentile = np.percentile(image, 1)
#     upper_percentile = np.percentile(image, 99)
#
#     histogram, _ = np.histogram(
#         image,
#         range=(lower_percentile, upper_percentile),
#         bins=128,
#         density=True
#     )
#
#     return histogram
#
#
# if __name__ == '__main__':
#     import SimpleITK as sitk
#     import matplotlib.pyplot as plt
#     for i in range(1, 4):
#         moving = sitk.ReadImage(f'/datalake/learn2reg/NLST/imagesTr/NLST_000{i}_0000.nii.gz')
#         moving = sitk.GetArrayFromImage(moving)
#         moving = np.clip(moving, -1024, 3071)
#
#         fixed = sitk.ReadImage(f'/datalake/learn2reg/NLST/imagesTr/NLST_000{i}_0001.nii.gz')
#         fixed = sitk.GetArrayFromImage(fixed)
#         fixed = np.clip(fixed, -1024, 3071)
#
#         l2 = (moving - fixed) ** 2
#
#         hm = extract_histogram_features(moving)
#         hf = extract_histogram_features(fixed)
#         hl2 = extract_histogram_features(l2)
#
#         plt.plot(hm - hf)
