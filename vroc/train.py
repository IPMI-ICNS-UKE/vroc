import matplotlib
import torch
import os
from glob import glob
from torch.utils.data import DataLoader

from vroc.models import TrainableVarRegBlock
from vroc.dataset import VrocDataset

matplotlib.use("module://backend_interagg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dirlab_case = 1
n_level = 2


def generate_train_list(root_dir, image_folder, mask_folder):
    image_path = os.path.join(root_dir, image_folder)
    mask_path = os.path.join(root_dir, mask_folder)
    flist = os.listdir(image_path)
    train_list = []
    for i in range(len(flist) // 2):
        fixed = os.path.join(image_path, f"NLST_{i:04d}_0000.nii.gz")
        moving = os.path.join(image_path, f"NLST_{i:04d}_0001.nii.gz")
        mask = os.path.join(mask_path, f"NLST_{i:04d}_0000.nii.gz")
        train_list.append({"fixed": fixed, "moving": moving, "mask": mask})
    return train_list


train_list = generate_train_list(
    root_dir="/home/tsentker/Documents/projects/learn2reg/NLST",
    image_folder="imagesTr",
    mask_folder="masksTr",
)
train_dataset = VrocDataset(dir_list=train_list)

dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

for data in dataloader:
    fixed = data["fixed"][0].to(device)
    mask = data["mask"][0].to(device)
    moving = data["moving"][0].to(device)
    model = TrainableVarRegBlock(
        patch_shape=data["patch_shape"][0],
        iterations=(1100,) * n_level,
        demon_forces="symmetric",
        tau=(6,) * n_level,
        regularization_sigma=((3.0, 3.0, 3.0),) * n_level,
        scale_factors=(1 / 2, 1 / 1),
        disable_correction=True,
        early_stopping_fn=("none", "lstsq"),
        early_stopping_delta=(10, 1e-3),
        early_stopping_window=(10, 20),
    ).to(device)
    _, warped, _, vf, metrics, features = model.forward(
        fixed, mask, moving, data["spacing"][0]
    )

loss = []
for m in metrics:
    l = (m[-1] - m[0]) / m[-1]
    loss.append(l)
