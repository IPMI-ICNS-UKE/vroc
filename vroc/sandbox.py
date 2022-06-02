import SimpleITK as sitk
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

from vroc.models import TrainableVarRegBlock
from vroc.helper import (
    read_landmarks,
    transform_landmarks_and_flip_z,
    target_registration_errors_snapped,
    load_and_preprocess,
    landmark_distance,
    scale_vf,
    target_registration_errors,
    detach_and_squeeze,
)

matplotlib.use("module://backend_interagg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dirlab_case = 1
n_level = 2

data_path = f"/home/tsentker/data/dirlab2022/data/Case{dirlab_case:02d}Pack"
out_path = "/home/tsentker/Documents/results/varreg-on-crack/"

fixed = load_and_preprocess(os.path.join(data_path, "Images", "phase_0.mha"))
orig_ref = fixed
original_image_spacing = fixed.GetSpacing()
moving = load_and_preprocess(os.path.join(data_path, "Images", "phase_5.mha"))
mask = sitk.ReadImage(os.path.join(data_path, "segmentation", f"mask_0.mha"))
moving = sitk.HistogramMatching(
    moving, fixed, numberOfHistogramLevels=1024, numberOfMatchPoints=7
)
fixed = sitk.GetArrayFromImage(fixed)
moving = sitk.GetArrayFromImage(moving)
mask = sitk.GetArrayFromImage(mask)

patch_shape = fixed.shape
fixed_ = torch.from_numpy(fixed.copy())
fixed_ = fixed_[None, None, :].float().to(device)
moving_ = torch.from_numpy(moving.copy())
moving_ = moving_[None, None, :].float().to(device)
mask_ = torch.from_numpy(mask.copy())
mask_ = mask_[None, None, :].float().to(device)
# mask_ = torch.ones_like(fixed_).float().to(device)

model = TrainableVarRegBlock(
    patch_shape=patch_shape,
    iterations=(1100,) * n_level,
    demon_forces="symmetric",
    tau=(6,) * n_level,
    regularization_sigma=((3.0, 3.0, 3.0),) * n_level,
    # radius=(4, 4, 4, 4),
    scale_factors=(1 / 2, 1 / 1),
    disable_correction=True,
    early_stopping_fn=("none", "lstsq"),
    early_stopping_delta=(10, 1e-3),
    early_stopping_window=(10, 20),
).to(device)
_, warped, _, vf, metrics, _ = model.forward(
    fixed_, mask_, moving_, original_image_spacing
)

warped_moving = detach_and_squeeze(warped)
warped_moving.CopyInformation(orig_ref)
sitk.WriteImage(warped_moving, os.path.join(out_path, "warped_moving.mha"))

vector_field = detach_and_squeeze(vf, is_vf=True)
vector_field = scale_vf(vector_field, spacing=orig_ref.GetSpacing())
vector_field.CopyInformation(orig_ref)

sigma_jac = sitk.DisplacementFieldJacobianDeterminant(vector_field, True)
sigma_jac = sitk.GetArrayFromImage(sigma_jac)[sitk.GetArrayFromImage(mask) == 1]

fixed_LM = read_landmarks(
    os.path.join(data_path, "extremePhases", f"Case{dirlab_case}_300_T00_xyz.txt")
)
fixed_LM_world = transform_landmarks_and_flip_z(fixed_LM, orig_ref)
moving_LM = read_landmarks(
    os.path.join(data_path, "extremePhases", f"Case{dirlab_case}_300_T50_xyz.txt")
)
moving_LM_world = transform_landmarks_and_flip_z(moving_LM, orig_ref)

delta_LM = landmark_distance(fixed_LM_world, moving_LM_world)

vf_transformed = sitk.DisplacementFieldTransform(vector_field)
TRE_world = target_registration_errors_snapped(
    vf_transformed, fixed_LM_world, moving_LM_world, orig_ref, world=True
)
TRE_voxel = target_registration_errors_snapped(
    vf_transformed, fixed_LM_world, moving_LM_world, orig_ref, world=False
)
TRE_world_no_snap = target_registration_errors(
    vf_transformed, fixed_LM_world, moving_LM_world
)

print("| Case |  voxel TRE  | world TRE[mm] | no snap world TRE[mm] | sigma jacobian |")
print(
    f"|  {dirlab_case:02d}  | {np.mean(TRE_voxel):.2f} ({np.std(TRE_voxel):.2f}) |  {np.mean(TRE_world):.2f} ({np.std(TRE_world):.2f})  |      {np.mean(TRE_world_no_snap):.2f} ({np.std(TRE_world_no_snap):.2f})      |      {np.std(sigma_jac):.2f}      |"
)
