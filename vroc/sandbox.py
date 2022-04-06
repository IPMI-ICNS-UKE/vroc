import SimpleITK as sitk
import os
import time

import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib

matplotlib.use('WebAgg')
from vroc.models import TrainableVarRegBlock

from vroc.helper import read_landmarks, transform_landmarks_and_flip_z, target_registration_errors, load_and_preprocess, \
    landmark_distance, plot_TRE_landmarks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dirlab_case = 1

data_path = f'/home/tsentker/data/dirlab2022/data/Case{dirlab_case:02d}Pack'
out_path = '/home/tsentker/Documents/results/varreg-on-crack/'

fixed = load_and_preprocess(os.path.join(data_path, 'Images', 'phase_0.mha'))
orig_ref = fixed
moving = load_and_preprocess(os.path.join(data_path, 'Images', 'phase_5.mha'))
mask = sitk.ReadImage(os.path.join(data_path, 'segmentation', f'mask_0.mha'))
# TODO: Influence of mask dilation on TRE
dilate_filter = sitk.BinaryDilateImageFilter()
dilate_filter.SetForegroundValue(1)
dilate_filter.SetKernelRadius((2, 2, 2))
mask = dilate_filter.Execute(mask)
moving = sitk.HistogramMatching(moving, fixed)
hist_matching = sitk.HistogramMatchingImageFilter()
hist_matching.SetNumberOfHistogramLevels(1024)
hist_matching.SetNumberOfMatchPoints(7)
hist_matching.ThresholdAtMeanIntensityOn()
hist_matching.Execute(moving, fixed)
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
start = time.time()
model = TrainableVarRegBlock(patch_shape=patch_shape,
                             iterations=(1600, 1600, 1600, 1600),
                             demon_forces='active',
                             tau=2.0,
                             regularization_sigma=((2.0, 2.0, 2.0), (2.0, 2.0, 2.0), (2.0, 2.0, 2.0), (2.0, 2.0, 2.0)),
                             scale_factors=(0.125, 0.25, 0.5, 1.0),
                             disable_correction=True,
                             early_stopping_delta=1e-3,
                             early_stopping_window=20).to(device)
_, warped, _, vf = model.forward(fixed_, mask_, moving_)
print(time.time() - start)

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

warper.SetOutputParameteresFromImage(moving)
sitk_warped_moving_scaled = warper.Execute(moving, vector_field_scaled)
sitk.WriteImage(sitk_warped_moving_scaled, os.path.join(out_path, 'sitk_warped_moving_scaled.mha'))
sitk.WriteImage(vector_field_scaled, os.path.join(out_path, 'sitk_vf.mha'))
vf_transformed = sitk.DisplacementFieldTransform(vector_field_scaled)

delta_LM = landmark_distance(fixed_LM_world, moving_LM_world)
TRE = target_registration_errors(vf_transformed, fixed_LM_world, moving_LM_world)
print(
    f'Mean (std) TRE for DIRlab case {dirlab_case} is {np.mean(TRE):.2f} ({np.std(TRE):.2f}) [before registration: {np.mean(delta_LM):.2f} ({np.std(delta_LM):.2f}) ]')
#
# plt.imshow(sitk.GetArrayFromImage(moving)[:,100,:], aspect='auto')
# # plt.show()
# plot_TRE_landmarks(vf_transformed,moving_LM, fixed_LM)
# plt.show()
