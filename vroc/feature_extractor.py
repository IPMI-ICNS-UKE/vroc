import os
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from dataset import AutoencoderDataset
from helper import rescale_range
from tqdm import tqdm

from vroc.dataset import BaseDataset
from vroc.models import AutoEncoder
from vroc.preprocessing import crop_background, resample_image_size


class FeatureExtractor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=torch.device(self.device))
        # self.model = AutoEncoder()
        # self.model.load_state_dict(torch.load(model_path)["state_dict"])
        self.model.eval()

    def infer(self, image: np.ndarray):
        image = self._prepare_image(image=image)
        image = self._prepare_torch_tensor(image)
        with torch.no_grad():
            _, embedded = self.model(image)
        return embedded

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        return rescale_range(image, input_range=(-1024, 3071), output_range=(0, 1))

    def _prepare_torch_tensor(self, image: np.ndarray) -> torch.tensor:
        image = torch.as_tensor(image.copy(), dtype=torch.float32, device=self.device)
        return image[None, None, :]


if __name__ == "__main__":
    feature_extractor = FeatureExtractor(
        model_path="/home/tsentker/Documents/results/vroc_AE/models/AE_100_epochs_luna16_NLST.pth"
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
