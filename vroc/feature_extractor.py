import os
import re
from pathlib import Path

import torch
from dataset import AutoencoderDataset
from tqdm import tqdm

from vroc.models import AutoEncoder


class FeatureExtractor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=torch.device(self.device))
        # self.model = AutoEncoder()
        # self.model.load_state_dict(torch.load(model_path)["state_dict"])
        self.model.eval()

    def infer(self, image_path):
        image, _ = AutoencoderDataset([image_path])[0]
        image = image[None, ...]
        with torch.no_grad():
            image = image.to(self.device)
            _, embedded = self.model(image)
        return embedded


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
        feature_vector = feature_extractor.infer(image_path=filepath)
        case = re.search(r"Case(\d*)Pack", str(filepath))[1]
        features[(case, os.path.splitext(os.path.basename(filepath))[0])] = (
            feature_vector.detach().cpu().numpy()
        )
