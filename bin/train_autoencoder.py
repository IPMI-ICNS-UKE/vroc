import os
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import trange

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
    train_list = []
    for dataset in dataset_paths:
        paths = glob(os.path.join(dataset, "*.nii.gz")) + glob(
            os.path.join(dataset, "*.mhd")
        )
        train_list.extend(paths)
    return train_list


data_path_list = generate_path_list(
    dataset_paths=[
        "/home/tsentker/data/learn2reg/NLST/imagesTr",
        "/home/tsentker/data/luna16/images",
    ]
)

train_list, val_list = train_test_split(
    data_path_list, random_state=seed, train_size=0.8, test_size=0.2
)

train_dataset = AutoEncoderDataset(dir_list=train_list)
val_dataset = AutoEncoderDataset(dir_list=val_list)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=16)


pbar = trange(1, n_epochs + 1)
epoch_losses = []
val_losses = []
val_loss = 0.0
for epoch in pbar:
    running_loss = 0.0

    for data, _ in train_loader:
        images = data.to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    in_img = images[0].squeeze().detach().cpu().numpy()
    out_img = outputs[0].squeeze().detach().cpu().numpy()

    in_img = sitk.GetImageFromArray(in_img)
    out_img = sitk.GetImageFromArray(out_img)

    sitk.WriteImage(
        in_img,
        f"/home/tsentker/Documents/results/vroc_AE/in_img_epoch{epoch:03d}.nii.gz",
    )
    sitk.WriteImage(
        out_img,
        f"/home/tsentker/Documents/results/vroc_AE/out_img_epoch{epoch:03d}.nii.gz",
    )

    epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(epoch_loss)

    if epoch % 5 == 0:
        val_loss = 0.0
        for data, _ in val_loader:
            with torch.no_grad():
                images = data.to(device)
                outputs, embedded = model(images)
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, images)
                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

    pbar.set_description(
        f"epoch: {epoch} \ttrain loss: {epoch_loss:.3f} \tval loss: {val_loss:.3f}"
    )
