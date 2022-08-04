import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
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

OUTPUT_FOLDER = Path("/datalake/learn2reg/NLST_Validation/predictions")


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

registration = VrocRegistration(
    roi_segmenter=None,
    feature_extractor=None,
    parameter_guesser=None,
    default_parameters=VrocRegistration.DEFAULT_REGISTRATION_PARAMETERS,
    debug=True,
    device="cuda:0",
)


FOLDER = "NLST_Validation"
# FOLDER = "NLST"

tres_before = []
tres_after = []
t_start = time.time()
for case in range(101, 111):
    fixed_landmarks = read_landmarks(
        f"/datalake/learn2reg/{FOLDER}/keypointsTr/NLST_{case:04d}_0000.csv",
        sep=" ",
    )
    moving_landmarks = read_landmarks(
        f"/datalake/learn2reg/{FOLDER}/keypointsTr/NLST_{case:04d}_0001.csv",
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
        moving_image_filepath=f"/datalake/learn2reg/{FOLDER}/imagesTr/NLST_{case:04d}_0001.nii.gz",
        fixed_image_filepath=f"/datalake/learn2reg//{FOLDER}/imagesTr/NLST_{case:04d}_0000.nii.gz",
        moving_mask_filepath=f"/datalake/learn2reg//{FOLDER}/masksTr/NLST_{case:04d}_0001.nii.gz",
        fixed_mask_filepath=f"/datalake/learn2reg//{FOLDER}/masksTr/NLST_{case:04d}_0000.nii.gz",
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
        early_stopping_delta=0.001,
        early_stopping_window=100,
    )

    # fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    # mid_slice = fixed_image.shape[1] // 2
    # clim = (-1000, 200)
    # ax[0, 0].imshow(fixed_image[:, mid_slice, :], clim=clim)
    # ax[0, 1].imshow(moving_image[:, mid_slice, :], clim=clim)
    # ax[0, 2].imshow(warped_image[:, mid_slice, :], clim=clim)
    #
    # ax[1, 0].imshow(vector_field[2, :, mid_slice, :], clim=(-10, 10), cmap="seismic")
    # ax[1, 0].set_title("VF")
    #
    # ax[1, 1].imshow(
    #     moving_image[:, mid_slice, :] - fixed_image[:, mid_slice, :],
    #     clim=(-500, 500),
    #     cmap="seismic",
    # )
    # ax[1, 2].imshow(
    #     warped_image[:, mid_slice, :] - fixed_image[:, mid_slice, :],
    #     clim=(-500, 500),
    #     cmap="seismic",
    # )
    # fig.suptitle(f"NLST_{case:04d}")

    # output_filepath = write_nlst_vector_field(
    #     vector_field,
    #     reference_image=reference_image,
    #     case=f"{case:04d}_test",
    #     output_folder=OUTPUT_FOLDER,
    # )

    vf = torch.as_tensor(vector_field[np.newaxis], device="cuda:0")
    ml = torch.as_tensor(moving_landmarks[np.newaxis], device="cuda:0")
    fl = torch.as_tensor(fixed_landmarks[np.newaxis], device="cuda:0")

    tre_loss = TRELoss(image_spacing=(1.5, 1.5, 1.5), apply_sqrt=True).to("cuda:0")
    loss_before = float(tre_loss(vf * 0.0, ml, fl))
    loss_after = float(tre_loss(vf, ml, fl))

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
        f"tre_loss_before={loss_before:.2f}, "
        f"tre_loss_after={loss_after:.2f}"
    )
    tres_before.append(np.mean(tre_before))
    tres_after.append(np.mean(tre_after))

print(f"before: mean TRE={np.mean(tres_before)}, std TRE={np.std(tres_before)}")
print(f"after: mean TRE={np.mean(tres_after)}, std TRE={np.std(tres_after)}")

print(f"run took {time.time() - t_start}")
