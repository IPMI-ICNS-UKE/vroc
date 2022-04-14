import SimpleITK as sitk
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

from vroc.models import TrainableVarRegBlock
from vroc.helper import read_landmarks, transform_landmarks_and_flip_z, target_registration_errors_snapped, \
    load_and_preprocess, landmark_distance, plot_TRE_landmarks, target_registration_errors

matplotlib.use('module://backend_interagg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dirlab_case = 8
n_level = 4

data_path = f'/home/tsentker/data/dirlab2022/data/Case{dirlab_case:02d}Pack'
out_path = '/home/tsentker/Documents/results/varreg-on-crack/'

fixed = load_and_preprocess(os.path.join(data_path, 'Images', 'phase_0.mha'))
orig_ref = fixed
original_image_spacing = fixed.GetSpacing()
moving = load_and_preprocess(os.path.join(data_path, 'Images', 'phase_5.mha'))
mask = sitk.ReadImage(os.path.join(data_path, 'segmentation', f'mask_0.mha'))
# TODO: Influence of mask dilation on TRE -> Further, use ebolagnul for mask prediction
dilate_filter = sitk.BinaryDilateImageFilter()
dilate_filter.SetForegroundValue(1)
dilate_filter.SetKernelRadius((1, 1, 1))
mask = dilate_filter.Execute(mask)
moving = sitk.HistogramMatching(moving, fixed, numberOfHistogramLevels=1024, numberOfMatchPoints=7)
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

model = TrainableVarRegBlock(patch_shape=patch_shape,
                             iterations=(800,) * n_level,
                             demon_forces='symmetric',
                             tau=(2.0,) * n_level,
                             regularization_sigma=((2.0, 2.0, 2.0),) * n_level,
                             radius=(4, 4, 4, 4),
                             scale_factors=(1 / 8, 1 / 4, 1 / 2, 1 / 1),
                             disable_correction=True,
                             early_stopping_fn=('no_impr', 'none', 'none', 'lstsq'),
                             early_stopping_delta=(10, 10, 10, 1e-5),
                             early_stopping_window=(10, 10, 10, 20)).to(device)
# start = time.time()
_, warped, _, vf, metrics = model.forward(fixed_, mask_, moving_, original_image_spacing)
# print('Elapsed time is ' time.time() - start)

warped_moving = warped.cpu().detach().numpy()
warped_moving = np.squeeze(warped_moving)
warped_moving = sitk.GetImageFromArray(warped_moving)
warped_moving.CopyInformation(orig_ref)
sitk.WriteImage(warped_moving, os.path.join(out_path, 'warped_moving.mha'))

vector_field = vf.cpu().detach().numpy()
vector_field = np.squeeze(vector_field)
vector_field = np.rollaxis(vector_field, 0, vector_field.ndim)
vector_field = vector_field[..., ::-1]
vector_field = sitk.GetImageFromArray(vector_field, isVector=True)
vector_field = sitk.Cast(vector_field, sitk.sitkVectorFloat64)
warper = sitk.WarpImageFilter()
moving_tmp = sitk.GetImageFromArray(moving)
sitk_warped_moving = warper.Execute(moving_tmp, vector_field)
sitk_warped_moving.CopyInformation(orig_ref)
sitk.WriteImage(sitk_warped_moving, os.path.join(out_path, 'sitk_warped_moving.mha'))

moving = sitk.GetImageFromArray(moving)
moving.CopyInformation(orig_ref)
fixed = sitk.GetImageFromArray(fixed)
fixed.CopyInformation(orig_ref)
mask = sitk.GetImageFromArray(mask)
mask.CopyInformation(orig_ref)
sitk.WriteImage(mask, os.path.join(out_path, 'mask.mha'))
sitk.WriteImage(moving, os.path.join(out_path, 'moving.mha'))
sitk.WriteImage(fixed, os.path.join(out_path, 'fixed.mha'))

fixed_LM = read_landmarks(os.path.join(data_path, 'extremePhases', f'Case{dirlab_case}_300_T00_xyz.txt'))
fixed_LM_world = transform_landmarks_and_flip_z(fixed_LM, orig_ref)
moving_LM = read_landmarks(os.path.join(data_path, 'extremePhases', f'Case{dirlab_case}_300_T50_xyz.txt'))
moving_LM_world = transform_landmarks_and_flip_z(moving_LM, orig_ref)

vector_field_scaled = sitk.Compose(
    [sitk.VectorIndexSelectionCast(vector_field, i) * sp for i, sp in enumerate(orig_ref.GetSpacing())])
vector_field_scaled = sitk.Cast(vector_field_scaled, sitk.sitkVectorFloat64)
vector_field_scaled.CopyInformation(orig_ref)
sigma_jac = sitk.DisplacementFieldJacobianDeterminant(vector_field_scaled, True)
sigma_jac = sitk.GetArrayFromImage(sigma_jac)[sitk.GetArrayFromImage(mask) == 1]

warper.SetOutputParameteresFromImage(moving)
sitk_warped_moving_scaled = warper.Execute(moving, vector_field_scaled)
sitk.WriteImage(sitk_warped_moving_scaled, os.path.join(out_path, 'sitk_warped_moving_scaled.mha'))
sitk.WriteImage(vector_field_scaled, os.path.join(out_path, 'sitk_vf.mha'))
vf_transformed = sitk.DisplacementFieldTransform(vector_field_scaled)

delta_LM = landmark_distance(fixed_LM_world, moving_LM_world)
TRE_world = target_registration_errors_snapped(vf_transformed, fixed_LM_world, moving_LM_world, orig_ref, world=True)
TRE_voxel = target_registration_errors_snapped(vf_transformed, fixed_LM_world, moving_LM_world, orig_ref, world=False)
TRE_world_no_snap = target_registration_errors(vf_transformed, fixed_LM_world, moving_LM_world)

# print(
#     f'Mean (std) world TRE for DIRlab case {dirlab_case} is {np.mean(TRE_world):.2f} ({np.std(TRE_world):.2f}) [mean voxel TRE: {np.mean(TRE_voxel):.2f} ({np.std(TRE_voxel):.2f}); mean world TRE no snap: {np.mean(TRE_world_no_snap):.2f} ({np.std(TRE_world_no_snap):.2f})]')
# print(
#     f'Mean (std) jacobian for DIRlab case {dirlab_case} is {np.std(sigma_jac):.2f}')

print('| Case |  voxel TRE  | world TRE[mm] | no snap world TRE[mm] | sigma jacobian |')
print(
    f'|  {dirlab_case:02d}  | {np.mean(TRE_voxel):.2f} ({np.std(TRE_voxel):.2f}) |  {np.mean(TRE_world):.2f} ({np.std(TRE_world):.2f})  |      {np.mean(TRE_world_no_snap):.2f} ({np.std(TRE_world_no_snap):.2f})      |      {np.std(sigma_jac):.2f}      |')
#
# plt.imshow(sitk.GetArrayFromImage(moving)[:,100,:], aspect='auto')
# # plt.show()
# plot_TRE_landmarks(vf_transformed, fixed_LM_world, moving_LM_world)
# plt.show()
plt.plot([item for sublist in metrics for item in sublist])
plt.show()
