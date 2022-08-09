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
        forces="dual",
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


import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from matplotlib.animation import FFMpegWriter
from scipy.ndimage.morphology import binary_dilation

from vroc.common_types import PathLike
from vroc.feature_extractor import OrientedHistogramFeatureExtrator
from vroc.guesser import ParameterGuesser
from vroc.helper import compute_tre_numpy, compute_tre_sitk, read_landmarks
from vroc.logger import LogFormatter
from vroc.loss import TRELoss
from vroc.registration import VrocRegistration

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(LogFormatter())
logging.basicConfig(handlers=[handler])

logging.getLogger(__name__).setLevel(logging.DEBUG)
logging.getLogger("vroc").setLevel(logging.DEBUG)
logging.getLogger("vroc.models.VarReg3d").setLevel(logging.INFO)

ROOT_DIR = (
    Path("/home/tsentker/data/learn2reg"),
    Path("/datalake/learn2reg"),
)
ROOT_DIR = next(p for p in ROOT_DIR if p.exists())
FOLDER = "NLST_Validation"

device = "cuda:0"

OUTPUT_FOLDER = Path(f"{ROOT_DIR}/{FOLDER}/predictions")


# def write_nlst_vector_field(vector_field, case: str, output_folder: Path):
#     vector_field = sitk.GetArrayFromImage(vector_field)
#     vector_field = np.rollaxis(vector_field, -1, 0)
#     vector_field = sitk.GetImageFromArray(vector_field, isVector=False)
#
#     output_filepath = output_folder / f"disp_{case}_{case}.nii.gz"
#     sitk.WriteImage(vector_field, str(output_filepath))
#
#     return str(output_filepath)


def write_nlst_vector_field(
    vector_field: np.ndarray,
    reference_image: sitk.Image,
    case: str,
    output_folder: Path,
):
    # TODO: make this working again and clean this up. Is this all necessary?
    vector_field = np.rollaxis(vector_field, 0, vector_field.ndim)
    vector_field = np.swapaxes(vector_field, 0, 2)
    vector_field = sitk.GetImageFromArray(vector_field, isVector=True)
    vector_field = sitk.Cast(vector_field, sitk.sitkVectorFloat64)
    transform = sitk.DisplacementFieldTransform(vector_field)

    f = sitk.TransformToDisplacementFieldFilter()
    f.SetSize(fixed_image.shape)
    f.SetOutputSpacing((1.0, 1.0, 1.0))
    final_transform_vf = f.Execute(transform)
    final_transform_vf.SetDirection(reference_image.GetDirection())

    vector_field = sitk.GetArrayFromImage(vector_field)
    vector_field = np.rollaxis(vector_field, -1, 0)
    vector_field = sitk.GetImageFromArray(vector_field, isVector=False)

    output_filepath = output_folder / f"disp_{case}_{case}.nii.gz"
    sitk.WriteImage(vector_field, str(output_filepath))

    return str(output_filepath)


def load(
    moving_image_filepath,
    fixed_image_filepath,
    moving_mask_filepath,
    fixed_mask_filepath,
):
    moving_image = sitk.ReadImage(moving_image_filepath)
    fixed_image = sitk.ReadImage(fixed_image_filepath)
    moving_mask = sitk.ReadImage(moving_mask_filepath)
    fixed_mask = sitk.ReadImage(fixed_mask_filepath)

    reference_image = fixed_image

    image_spacing = fixed_image.GetSpacing()[::-1]

    moving_image = sitk.GetArrayFromImage(moving_image)
    fixed_image = sitk.GetArrayFromImage(fixed_image)
    moving_mask = sitk.GetArrayFromImage(moving_mask)
    fixed_mask = sitk.GetArrayFromImage(fixed_mask)

    moving_image = np.swapaxes(moving_image, 0, 2)
    fixed_image = np.swapaxes(fixed_image, 0, 2)
    moving_mask = np.swapaxes(moving_mask, 0, 2)
    fixed_mask = np.swapaxes(fixed_mask, 0, 2)

    return (
        moving_image,
        fixed_image,
        moving_mask,
        fixed_mask,
        image_spacing,
        reference_image,
    )


# feature_extractor = OrientedHistogramFeatureExtrator(device="cuda:0")
# parameter_guesser = ParameterGuesser(
#     database_filepath="/datalake/learn2reg/best_parameters.sqlite",
#     parameters_to_guess=('sigma_x', 'sigma_y', 'sigma_z')
# )
# parameter_guesser.fit()

params = {
    "iterations": 2000,
    "tau": 22.5,
    "sigma_x": 1.25,
    "sigma_y": 1.25,
    "sigma_z": 1.25,
    "n_levels": 3,
}

registration = VrocRegistration(
    roi_segmenter=None,
    feature_extractor=None,
    parameter_guesser=None,
    default_parameters=params,
    debug=False,
    device="cuda:0",
)

tres_before = []
tres_after = []
t_start = time.time()
for case in range(101, 111):
    fixed_landmarks = read_landmarks(
        f"{ROOT_DIR}/{FOLDER}/keypointsTr/NLST_{case:04d}_0000.csv",
        sep=" ",
    )
    moving_landmarks = read_landmarks(
        f"{ROOT_DIR}/{FOLDER}/keypointsTr/NLST_{case:04d}_0001.csv",
        sep=" ",
    )

    (
        moving_image,
        fixed_image,
        moving_mask,
        fixed_mask,
        image_spacing,
        reference_image,
    ) = load(
        moving_image_filepath=f"{ROOT_DIR}/{FOLDER}/imagesTr/NLST_{case:04d}_0001.nii.gz",
        fixed_image_filepath=f"{ROOT_DIR}/{FOLDER}/imagesTr/NLST_{case:04d}_0000.nii.gz",
        moving_mask_filepath=f"{ROOT_DIR}/{FOLDER}/masksTr/NLST_{case:04d}_0001.nii.gz",
        fixed_mask_filepath=f"{ROOT_DIR}/{FOLDER}/masksTr/NLST_{case:04d}_0000.nii.gz",
    )

    moving_mask = binary_dilation(moving_mask.astype(np.uint8), iterations=1).astype(
        np.uint8
    )
    fixed_mask = binary_dilation(fixed_mask.astype(np.uint8), iterations=1).astype(
        np.uint8
    )

    union_mask = moving_mask | fixed_mask

    warped_image, vector_field = registration.register(
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_mask=union_mask,
        fixed_mask=union_mask,
        register_affine=True,
        valid_value_range=(-1024, 3071),
        early_stopping_delta=1e-6,
        early_stopping_window=1600,
    )

    vf = torch.as_tensor(vector_field[np.newaxis], device="cuda:0")
    ml = torch.as_tensor(moving_landmarks[np.newaxis], device="cuda:0")
    fl = torch.as_tensor(fixed_landmarks[np.newaxis], device="cuda:0")

    tre_before = compute_tre_numpy(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=None,
        image_spacing=(1.5,) * 3,
    )
    tre_after = compute_tre_numpy(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=vector_field,
        image_spacing=(1.5,) * 3,
    )

    print(
        f"NLST_0{case}: "
        f"tre_before={np.mean(tre_before):.2f}, "
        f"tre_after={np.mean(tre_after):.2f}, "
    )
    tres_before.append(np.mean(tre_before))
    tres_after.append(np.mean(tre_after))

print(f"before: mean TRE={np.mean(tres_before)}, std TRE={np.std(tres_before)}")
print(f"after: mean TRE={np.mean(tres_after)}, std TRE={np.std(tres_after)}")

print(f"run took {time.time() - t_start}")
