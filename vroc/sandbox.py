import os

import matplotlib
import numpy as np
import SimpleITK as sitk
import torch

from vroc.dataset import BaseDataset
from vroc.helper import (
    compute_tre_numpy,
    compute_tre_sitk,
    detach_and_squeeze,
    read_landmarks,
    scale_vf,
)
from vroc.models import VarReg3d

matplotlib.use("module://backend_interagg")


def run_DIRLAB_registration(dirlab_case, p_fixed=0, p_moving=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dirlab_case = dirlab_case
    n_level = 2

    # DIRLAB 4DCT
    data_path = f"/home/tsentker/data/dirlab2022/data/Case{dirlab_case:02d}Pack"
    # out_path = "/home/tsentker/Documents/results/varreg-on-crack/"

    fixed = BaseDataset.load_and_preprocess(
        os.path.join(data_path, "Images", f"phase_{p_fixed}.mha")
    )
    orig_ref = fixed
    original_image_spacing = fixed.GetSpacing()
    moving = BaseDataset.load_and_preprocess(
        os.path.join(data_path, "Images", f"phase_{p_moving}.mha")
    )
    mask = sitk.ReadImage(
        os.path.join(data_path, "segmentation", f"mask_{p_fixed}.mha")
    )
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

    fixed_ = torch.from_numpy(fixed.copy())
    fixed_ = fixed_[None, None, :].float().to(device)
    moving_ = torch.from_numpy(moving.copy())
    moving_ = moving_[None, None, :].float().to(device)
    mask_ = torch.from_numpy(mask.copy())
    mask_ = mask_[None, None, :].float().to(device)

    model = VarReg3d(
        iterations=(100,) * n_level,
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

    # swap axis back to sitk format, detach and scale vf
    vector_field = detach_and_squeeze(vf, is_vf=True)
    vector_field = scale_vf(vector_field, spacing=orig_ref.GetSpacing())
    vector_field.CopyInformation(orig_ref)
    # #
    # # sigma_jac = sitk.DisplacementFieldJacobianDeterminant(vector_field, True)
    # # sigma_jac = sitk.GetArrayFromImage(sigma_jac)[sitk.GetArrayFromImage(mask) == 1]
    #
    fixed_LM = read_landmarks(
        os.path.join(data_path, "extremePhases", f"landmarks_{p_fixed}.txt")
    )
    moving_LM = read_landmarks(
        os.path.join(data_path, "extremePhases", f"landmarks_{p_moving}.txt")
    )

    vf_transformed = sitk.DisplacementFieldTransform(vector_field)

    vf = vf.cpu().detach().numpy()
    vf = np.squeeze(vf)

    tre_numpy = compute_tre_numpy(
        fixed_landmarks=np.array(fixed_LM),
        moving_landmarks=np.array(moving_LM),
        vector_field=vf,
        image_spacing=orig_ref.GetSpacing(),
        snap_to_voxel=True,
    )
    tre_sitk = compute_tre_sitk(
        fix_lms=np.array(fixed_LM),
        mov_lms=np.array(moving_LM),
        transform=vf_transformed,
        ref_img=orig_ref,
        spacing_mov=orig_ref.GetSpacing(),
        snap_to_voxel=True,
    )

    return tre_numpy, tre_sitk


def compute_TRE(reference_img_path, vf_path, lm_fixed_path, lm_moving_path):
    ref_img = sitk.ReadImage(reference_img_path)
    vf = sitk.ReadImage(vf_path, sitk.sitkVectorFloat64)
    vf = sitk.Resample(
        vf, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, vf.GetPixelID()
    )

    vf_arr = sitk.GetArrayFromImage(vf)
    vf_arr = np.swapaxes(vf_arr, 0, 2)
    vf_arr = np.moveaxis(vf_arr, 3, 0)
    vf_arr[0] /= ref_img.GetSpacing()[0]
    vf_arr[1] /= ref_img.GetSpacing()[1]
    vf_arr[2] /= ref_img.GetSpacing()[2]

    fixed_LM = read_landmarks(lm_fixed_path)
    moving_LM = read_landmarks(lm_moving_path)

    tre_np = compute_tre_numpy(
        fixed_landmarks=fixed_LM,
        moving_landmarks=moving_LM,
        vector_field=vf_arr,
        image_spacing=ref_img.GetSpacing(),
        snap_to_voxel=False,
    )

    transform = sitk.DisplacementFieldTransform(vf)
    tre_sitk = compute_tre_sitk(
        fix_lms=fixed_LM,
        mov_lms=moving_LM,
        transform=transform,
        ref_img=ref_img,
        spacing_mov=ref_img.GetSpacing(),
        snap_to_voxel=False,
    )

    return tre_np, tre_sitk


# tre_np, tre_sitk = compute_TRE(
#     reference_img_path=f'/home/tsentker/data/dirlab/CT_asMHD/Case{case:02d}/T00-Case{case:02d}.nii.gz',
#     vf_path=f'/home/tsentker/Downloads/case_{case:02d}_dirlab_hering_disp.mha',
#     lm_fixed_path=f'/home/tsentker/data/dirlab2022/data/Case{case:02d}Pack/extremePhases/Case{case}_300_T00_xyz.txt',
#     lm_moving_path=f'/home/tsentker/data/dirlab2022/data/Case{case:02d}Pack/extremePhases/Case{case}_300_T50_xyz.txt')

tre_numpy, tre_sitk = run_DIRLAB_registration(dirlab_case=2)
