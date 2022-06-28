import os
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from vroc.autoencoder_gym import AutoencoderGym
from vroc.dataset import AutoencoderDataset

seed = 1337

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = (
    Path("/home/tsentker/data"),
    Path("/datalake/NLST"),
)
root_dir = next(p for p in root_dir if p.exists())

dataset_paths = [
    "NLST/imagesTr",
    "luna16/images",
]
dataset_paths = [os.path.join(root_dir, path) for path in dataset_paths]

filepaths = AutoencoderDataset.fetch_filepaths(dataset_paths)

train_list, val_list = train_test_split(
    filepaths, random_state=seed, train_size=0.8, test_size=0.2
)

train_dataset = AutoencoderDataset(filepaths=train_list)
val_dataset = AutoencoderDataset(filepaths=val_list)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

Gym = AutoencoderGym(
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    out_path="/home/tsentker/Documents/results/vroc_AE/models",
)
Gym.workout(validation_epoch=5, intermediate_save=True)
