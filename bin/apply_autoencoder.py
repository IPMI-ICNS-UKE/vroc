import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from vroc.dataset import AutoEncoderDataset
from vroc.models import AutoEncoder

seed = 1337

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = (
    Path("/home/tsentker/data"),
    Path("/datalake/NLST"),
)
root_dir = next(p for p in root_dir if p.exists())

n_epochs = 100

model = AutoEncoder().to(device=device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def generate_path_list(dataset_paths):
    path_list = []
    for dataset in dataset_paths:
        paths = glob(os.path.join(dataset, "*.nii.gz")) + glob(
            os.path.join(dataset, "*.mhd")
        )
        path_list.extend(paths)
    return path_list


data_path_list = generate_path_list(
    dataset_paths=[
        "/home/tsentker/data/learn2reg/NLST/imagesTr",
        "/home/tsentker/data/luna16/images",
    ]
)

model = torch.load(
    "/home/tsentker/Documents/results/vroc_AE/models/AE_100_epochs_luna16_NLST.pth"
)
dataset = AutoEncoderDataset(dir_list=data_path_list)
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16)


features = {}
for data, path in tqdm(loader):
    with torch.no_grad():
        images = data.to(device)
        _, embedded = model(images)
        features[os.path.basename(path[0])] = embedded.detach().cpu().numpy()
