import SimpleITK as sitk
import os
import time

import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib

matplotlib.use('WebAgg')
from vroc.models import TrainableVarRegBlock

from vroc.helper import read_landmarks, transform_landmarks, target_registration_errors, load_and_preprocess, \
    landmark_distance, plot_TRE_landmarks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dirlab_case = 2

data_path = f'/home/tsentker/data/dirlab2022/data/Case0{dirlab_case}Pack'
out_path = '/home/tsentker/Documents/results/varreg-on-crack/'

# data_path = '/mnt/c/Users/Thilo/Documents/dirlab2022/data/Case06Pack'
# out_path = '/mnt/c/Users/Thilo/Desktop'

fixed = load_and_preprocess(os.path.join(data_path, 'Images', 'phase_0.mha'))
orig_ref = fixed
moving = load_and_preprocess(os.path.join(data_path, 'Images', 'phase_5.mha'))
mask = sitk.ReadImage(os.path.join(data_path, 'segmentation', f'T00-Case{dirlab_case:02d}_Lungs.mha'))
dilate_filter = sitk.BinaryDilateImageFilter()
dilate_filter.SetForegroundValue(1)
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
                             tau=1.0,
                             regularization_sigma=((2.0, 2.0, 2.0), (2.0, 2.0, 2.0), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5)),
                             scale_factors=(0.125, 0.25, 0.5, 1.0),
                             disable_correction=True,
                             early_stopping_delta=1e-8,
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
fixed_LM_world = transform_landmarks(fixed_LM, orig_ref)
moving_LM = read_landmarks(os.path.join(data_path, 'extremePhases', f'Case{dirlab_case}_300_T50_xyz.txt'))
moving_LM_world = transform_landmarks(moving_LM, orig_ref)


# TODO: Warp moving with scaled vector field on correct spacing/origin/direction to verify correctness of transformation

vector_field_scaled = sitk.GetArrayFromImage(vector_field)
vector_field_scaled = vector_field_scaled * orig_ref.GetSpacing()
vector_field_scaled = sitk.GetImageFromArray(vector_field_scaled, isVector=True)
vector_field_scaled = sitk.Cast(vector_field_scaled, sitk.sitkVectorFloat64)
vector_field_scaled.CopyInformation(orig_ref)
vf_transformed = sitk.DisplacementFieldTransform(vector_field_scaled)
sitk.WriteTransform(vf_transformed, os.path.join(out_path, 'sitk_transform.tfm'))
delta_LM = landmark_distance(moving_LM_world, fixed_LM_world)
TRE = target_registration_errors(vf_transformed, moving_LM_world, fixed_LM_world)
#
# plt.imshow(sitk.GetArrayFromImage(moving)[:,256,:], aspect='auto')
# # plt.show()
# plot_TRE_landmarks(vf_transformed,moving_LM, fixed_LM)
# plt.show()
