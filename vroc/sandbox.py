import os

import matplotlib
import numpy as np
import SimpleITK as sitk
import torch

from vroc.dataset import BaseDataset
from vroc.helper import compute_tre, read_landmarks
from vroc.models import TrainableVarRegBlock

matplotlib.use("module://backend_interagg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dirlab_case = 2
n_level = 2

data_path = f"/home/tsentker/data/dirlab2022/data/Case{dirlab_case:02d}Pack"
out_path = "/home/tsentker/Documents/results/varreg-on-crack/"

fixed = BaseDataset.load_and_preprocess(
    os.path.join(data_path, "Images", "phase_0.mha")
)
orig_ref = fixed
original_image_spacing = fixed.GetSpacing()
moving = BaseDataset.load_and_preprocess(
    os.path.join(data_path, "Images", "phase_5.mha")
)
mask = sitk.ReadImage(os.path.join(data_path, "segmentation", f"mask_0.mha"))
moving = sitk.HistogramMatching(
    moving, fixed, numberOfHistogramLevels=1024, numberOfMatchPoints=7
)

# load to numpy and swap back to x,y,z orientation
fixed = sitk.GetArrayFromImage(fixed)
fixed = np.swapaxes(fixed, 0, 2)
moving = sitk.GetArrayFromImage(moving)
moving = np.swapaxes(moving, 0, 2)
mask = sitk.GetArrayFromImage(mask)
mask = np.swapaxes(mask, 0, 2)

patch_shape = fixed.shape
fixed_ = torch.from_numpy(fixed.copy())
fixed_ = fixed_[None, None, :].float().to(device)
moving_ = torch.from_numpy(moving.copy())
moving_ = moving_[None, None, :].float().to(device)
mask_ = torch.from_numpy(mask.copy())
mask_ = mask_[None, None, :].float().to(device)

model = TrainableVarRegBlock(
    iterations=(800,) * n_level,
    demon_forces="symmetric",
    tau=(2,) * n_level,
    regularization_sigma=((2.0, 2.0, 2.0),) * n_level,
    scale_factors=(1 / 2, 1 / 1),
).to(device)
warped, vf, _ = model.forward(fixed_, mask_, moving_, original_image_spacing)

# warped_moving = detach_and_squeeze(warped)
# warped_moving.CopyInformation(orig_ref)
# # sitk.WriteImage(warped_moving, os.path.join(out_path, "warped_moving.mha"))
#
# vector_field = detach_and_squeeze(vf, is_vf=True)
# vector_field = scale_vf(vector_field, spacing=orig_ref.GetSpacing())
# vector_field.CopyInformation(orig_ref)
#
# # sigma_jac = sitk.DisplacementFieldJacobianDeterminant(vector_field, True)
# # sigma_jac = sitk.GetArrayFromImage(sigma_jac)[sitk.GetArrayFromImage(mask) == 1]
#
fixed_LM = read_landmarks(
    os.path.join(data_path, "extremePhases", f"Case{dirlab_case}_300_T00_xyz.txt")
)
moving_LM = read_landmarks(
    os.path.join(data_path, "extremePhases", f"Case{dirlab_case}_300_T50_xyz.txt")
)

# Fixes DIRLAB z orientation in landmarks by flipping last dimension
fixed_LM = [(p[0], p[1], orig_ref.GetSize()[2] - p[2]) for p in fixed_LM]
moving_LM = [(p[0], p[1], orig_ref.GetSize()[2] - p[2]) for p in moving_LM]

vf = vf.cpu().detach().numpy()
vf = np.squeeze(vf)
TRE_numpy = compute_tre(
    fix_lms=np.array(fixed_LM),
    mov_lms=np.array(moving_LM),
    disp=vf,
    spacing_mov=orig_ref.GetSpacing(),
    snap_to_voxel=True,
)
