import logging
import os
from glob import glob
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from vroc.dataset import VrocDataset

# from vroc.metrics import calculate_mutual_information
from vroc.models import TrainableVarRegBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dirlab_case = 1
n_levels = 3
LOG_LEVEL = logging.DEBUG

logging.basicConfig(level=LOG_LEVEL)


def generate_train_list(root_dir, image_folder, mask_folder):
    image_path = os.path.join(root_dir, image_folder)
    mask_path = os.path.join(root_dir, mask_folder)
    flist = os.listdir(image_path)
    train_list = []
    for i in range(1, len(flist) // 2 + 1):
        fixed = os.path.join(image_path, f"NLST_{i:04d}_0000.nii.gz")
        moving = os.path.join(image_path, f"NLST_{i:04d}_0001.nii.gz")
        mask = os.path.join(mask_path, f"NLST_{i:04d}_0000.nii.gz")
        train_list.append({"fixed": fixed, "moving": moving, "mask": mask})
    return train_list[:1]


root_dir = (
    Path("/home/tsentker/Documents/projects/learn2reg/NLST"),
    Path("/datalake/NLST"),
)
root_dir = next(p for p in root_dir if p.exists())

train_list = generate_train_list(
    root_dir=root_dir,
    image_folder="imagesTr",
    mask_folder="masksTr",
)
train_dataset = VrocDataset(dir_list=train_list)

dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

scale_factors = tuple(1 / 2**i_level for i_level in reversed(range(n_levels)))

model = TrainableVarRegBlock(
    iterations=(100,) * n_levels,
    demon_forces="symmetric",
    tau=(6,) * n_levels,
    regularization_sigma=((3.0, 3.0, 3.0),) * n_levels,
    scale_factors=scale_factors,
    disable_correction=True,
    early_stopping_method=None,
    # early_stopping_delta=(10, 1e-3),
    # early_stopping_window=N(10, 20),
).to(device)

loss = []
for data in dataloader:
    fixed = data["fixed"][0].to(device)
    mask = data["mask"][0].to(device)
    moving = data["moving"][0].to(device)

    f = fixed.cpu().numpy()
    m = moving.cpu().numpy()

    hist2d, _, _ = np.histogram2d(f.ravel(), m.ravel(), bins=128)
    mi = calculate_mutual_information(f, m, bins=128)

    # print('go')
    # with torch.no_grad():
    #     warped_moving_image, vector_field, misc = model.forward(
    #         fixed, mask, moving, data["spacing"][0]
    #     )
