import SimpleITK as sitk
import os
import torch
import numpy as np
from hyperopt import tpe, hp, fmin

from vroc.models import TrainableVarRegBlock
from vroc.helper import (
    read_landmarks,
    transform_landmarks_and_flip_z,
    target_registration_errors_snapped,
    load_and_preprocess,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dirlab_cases = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ref_imgs = []
fixed_imgs = []
moving_imgs = []
mask_imgs = []
fixed_imgs_LM = []
moving_imgs_LM = []

for case in dirlab_cases:
    data_path = f"/home/tsentker/data/dirlab2022/data/Case{case:02d}Pack"
    fixed = load_and_preprocess(os.path.join(data_path, "Images", "phase_0.mha"))
    orig_ref = fixed
    ref_imgs.append(orig_ref)
    # original_image_spacing = fixed.GetSpacing()
    moving = load_and_preprocess(os.path.join(data_path, "Images", "phase_5.mha"))
    mask = sitk.ReadImage(os.path.join(data_path, "segmentation", f"mask_0.mha"))
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetForegroundValue(1)
    dilate_filter.SetKernelRadius((1, 1, 1))
    mask = dilate_filter.Execute(mask)
    moving = sitk.HistogramMatching(
        moving, fixed, numberOfHistogramLevels=1024, numberOfMatchPoints=7
    )
    fixed = sitk.GetArrayFromImage(fixed)
    moving = sitk.GetArrayFromImage(moving)
    mask = sitk.GetArrayFromImage(mask)

    patch_shape = fixed.shape
    fixed_ = torch.from_numpy(fixed.copy())
    fixed_imgs.append(fixed_[None, None, :].float().to(device))
    moving_ = torch.from_numpy(moving.copy())
    moving_imgs.append(moving_[None, None, :].float().to(device))
    mask_ = torch.from_numpy(mask.copy())
    mask_imgs.append(mask_[None, None, :].float().to(device))

    fixed_LM = read_landmarks(
        os.path.join(data_path, "extremePhases", f"Case{case}_300_T00_xyz.txt")
    )
    fixed_imgs_LM.append(transform_landmarks_and_flip_z(fixed_LM, orig_ref))
    moving_LM = read_landmarks(
        os.path.join(data_path, "extremePhases", f"Case{case}_300_T50_xyz.txt")
    )
    moving_imgs_LM.append(transform_landmarks_and_flip_z(moving_LM, orig_ref))

search_space = {
    "iter": hp.randint("iter", 500, 1500),
    "level": hp.randint("level", 2, 4),
    "tau": hp.uniform("tau", 1.0, 3.0),
    "sigma_x": hp.uniform("sigma_x", 1.0, 4.0),
    "sigma_y": hp.uniform("sigma_y", 1.0, 4.0),
    "sigma_z": hp.uniform("sigma_z", 1.0, 4.0),
    # 'radius': hp.randint('radius', 2, 5),
    # 'early_stopping_delta': hp.uniform('early_stopping_delta', 1e-6, 1e-4),
}


def reg(params):
    scale_factors = tuple(1 / 2**n for n in reversed(range(params["level"])))
    early_stopping_fn = ["none"] * params["level"]
    early_stopping_fn[-1] = "lstsq"

    early_stopping_delta = [10.0] * params["level"]
    early_stopping_delta[-1] = 1e-3  # params['early_stopping_delta']

    TRE_list = []
    for i, _ in enumerate(dirlab_cases):
        model = TrainableVarRegBlock(
            patch_shape=sitk.GetArrayFromImage(ref_imgs[i]).shape,
            iterations=(params["iter"],) * params["level"],
            demon_forces="symmetric",
            tau=(params["tau"],) * params["level"],
            regularization_sigma=(
                (params["sigma_x"], params["sigma_y"], params["sigma_z"]),
            )
            * params["level"],
            # radius=(params['radius'],) * params['level'],
            scale_factors=scale_factors,
            disable_correction=True,
            early_stopping_method=early_stopping_fn,
            early_stopping_delta=early_stopping_delta,
            early_stopping_window=(20,) * params["level"],
        ).to(device)
        _, _, _, vf, metrics = model.forward(
            fixed_imgs[i], mask_imgs[i], moving_imgs[i], ref_imgs[i].GetSpacing()
        )

        vector_field = vf.cpu().detach().numpy()
        vector_field = np.squeeze(vector_field)
        vector_field = np.rollaxis(vector_field, 0, vector_field.ndim)
        vector_field = vector_field[..., ::-1]
        vector_field = sitk.GetImageFromArray(vector_field, isVector=True)
        vector_field = sitk.Cast(vector_field, sitk.sitkVectorFloat64)

        vector_field_scaled = sitk.Compose(
            [
                sitk.VectorIndexSelectionCast(vector_field, i) * sp
                for i, sp in enumerate(ref_imgs[i].GetSpacing())
            ]
        )
        vector_field_scaled.CopyInformation(ref_imgs[i])
        vf_transformed = sitk.DisplacementFieldTransform(vector_field_scaled)
        TRE_world = target_registration_errors_snapped(
            vf_transformed, fixed_imgs_LM[i], moving_imgs_LM[i], ref_imgs[i], world=True
        )
        TRE_list.append(np.mean(TRE_world))

    return np.mean(TRE_list)


best = fmin(
    fn=reg,
    space=search_space,
    algo=tpe.suggest,
    max_evals=100000,
    trials_save_file="/home/tsentker/Documents/projects/varreg-on-crack/hyperopt_results_all_cases_small_tau.pkl",
)
