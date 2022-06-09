import matplotlib
import torch
import os
from glob import glob
from torch.utils.data import DataLoader

from vroc.models import TrainableVarRegBlock
from vroc.dataset import VrocDataset
from pathlib import Path

# matplotlib.use("module://backend_interagg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dirlab_case = 1
n_level = 1


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
    Path('/home/tsentker/Documents/projects/learn2reg/NLST'),
    Path('/datalake/NLST'),
)
root_dir = next(p for p in root_dir if p.exists())

train_list = generate_train_list(
    root_dir=root_dir,
    image_folder="imagesTr",
    mask_folder="masksTr",
)
train_dataset = VrocDataset(dir_list=train_list)

dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

model = TrainableVarRegBlock(
    iterations=(21,) * n_level,
    demon_forces="symmetric",
    tau=(6,) * n_level,
    regularization_sigma=((3.0, 3.0, 3.0),) * n_level,
    scale_factors=(0.5,), #(1 / 2, 1 / 1),
    disable_correction=True,
    early_stopping_fn=("none", "none"),
    # early_stopping_delta=(10, 1e-3),
    # early_stopping_window=N(10, 20),
).to(device)

loss = []
for data in dataloader:
    fixed = data["fixed"][0].to(device)
    mask = data["mask"][0].to(device)
    moving = data["moving"][0].to(device)

    for i in range(100000):
        print('go')
        model.forward(
            fixed, mask, moving, data["spacing"][0]
        )
        pass

