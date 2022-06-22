import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from vroc.dataset import AutoEncoderDataset
from vroc.models import AutoEncoder

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


def generate_train_list(dataset_paths):
    train_list = []
    for dataset in dataset_paths:
        paths = [os.path.join(dataset, f) for f in os.listdir(dataset)]
        train_list.extend(paths)
    return train_list


train_list = generate_train_list(
    dataset_paths=[
        "/home/tsentker/data/learn2reg/NLST/imagesTr",
    ]
)
train_dataset = AutoEncoderDataset(dir_list=train_list)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16)

pbar = trange(1, n_epochs + 1)
running_train_loss = 0.0
for epoch in pbar:
    train_loss = 0.0
    pbar.set_description(f"epoch: {epoch} \ttrain loss: {running_train_loss:.3f}")

    for data in train_loader:
        images = data.to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

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

    running_train_loss = train_loss / len(train_loader)
