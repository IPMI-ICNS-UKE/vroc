import matplotlib
import torch
from torch.utils.data import DataLoader

from vroc.models import TrainableVarRegBlock
from vroc.dataset import VrocDataset

matplotlib.use("module://backend_interagg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dirlab_case = 1
n_level = 2

train_list = [
    {
        "fixed": f"/home/tsentker/data/dirlab2022/data/Case{dirlab_case:02d}Pack/Images/phase_0.mha",
        "mask": f"/home/tsentker/data/dirlab2022/data/Case{dirlab_case:02d}Pack/segmentation/mask_0.mha",
        "moving": f"/home/tsentker/data/dirlab2022/data/Case{dirlab_case:02d}Pack/Images/phase_5.mha",
    }
]

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
