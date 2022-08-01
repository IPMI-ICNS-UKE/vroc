import logging
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
from vroc.helper import compute_tre, compute_tre_sitk, read_landmarks
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


def write_vector_field(vector_field, case: str, output_folder: Path):
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

    f = sitk.ClampImageFilter()
    f.SetUpperBound(2071)

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
    registration_parameters=VrocRegistration.DEFAULT_REGISTRATION_PARAMETERS,
    debug=True,
    device="cuda:0",
)


tres_before = []
tres_after = []
for case in range(101, 111):
    fixed_landmarks = read_landmarks(
        f"/datalake/learn2reg/NLST_Validation/keypointsTr/NLST_0{case}_0000.csv",
        sep=" ",
    )
    moving_landmarks = read_landmarks(
        f"/datalake/learn2reg/NLST_Validation/keypointsTr/NLST_0{case}_0001.csv",
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
        moving_image_filepath=f"/datalake/learn2reg/NLST_Validation/imagesTr/NLST_0{case}_0001.nii.gz",
        fixed_image_filepath=f"/datalake/learn2reg/NLST_Validation/imagesTr/NLST_0{case}_0000.nii.gz",
        moving_mask_filepath=f"/datalake/learn2reg/NLST_Validation/masksTr/NLST_0{case}_0001.nii.gz",
        fixed_mask_filepath=f"/datalake/learn2reg/NLST_Validation/masksTr/NLST_0{case}_0000.nii.gz",
    )

    union_mask = moving_mask | fixed_mask

    union_mask = binary_dilation(union_mask.astype(np.uint8), iterations=1).astype(
        np.uint8
    )

    moving_mask = union_mask
    fixed_mask = union_mask
    # moving_mask = binary_dilation(moving_mask.astype(np.uint8), iterations=1).astype(
    #     np.uint8
    # )
    # fixed_mask = binary_dilation(fixed_mask.astype(np.uint8), iterations=1).astype(
    #     np.uint8
    # )

    warped_image, transforms = registration.register(
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_mask=moving_mask,
        fixed_mask=fixed_mask,
        register_affine=True,
        valid_value_range=(-1024, 3071),
    )

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    mid_slice = fixed_image.shape[1] // 2
    clim = (-1000, 200)
    ax[0, 0].imshow(fixed_image[:, mid_slice, :], clim=clim)
    ax[0, 1].imshow(moving_image[:, mid_slice, :], clim=clim)
    ax[0, 2].imshow(warped_image[:, mid_slice, :], clim=clim)

    ax[1, 0].imshow(transforms[-1][2, :, mid_slice, :], clim=(-10, 10), cmap="seismic")
    ax[1, 0].set_title("VF")

    ax[1, 1].imshow(
        moving_image[:, mid_slice, :] - fixed_image[:, mid_slice, :],
        clim=(-500, 500),
        cmap="seismic",
    )
    ax[1, 2].imshow(
        warped_image[:, mid_slice, :] - fixed_image[:, mid_slice, :],
        clim=(-500, 500),
        cmap="seismic",
    )
    fig.suptitle(f"NLST_0{case}")

    vf = transforms[-1]
    vf = np.rollaxis(vf, 0, vf.ndim)
    vf = np.swapaxes(vf, 0, 2)
    vector_field = sitk.GetImageFromArray(vf, isVector=True)
    vector_field = sitk.Cast(vector_field, sitk.sitkVectorFloat64)
    transform = sitk.DisplacementFieldTransform(vector_field)

    if len(transforms) > 1:
        final_transform = sitk.CompositeTransform([transforms[0], transform])
    else:
        final_transform = transform

    f = sitk.TransformToDisplacementFieldFilter()
    f.SetSize(fixed_image.shape)
    f.SetOutputSpacing((1.0, 1.0, 1.0))
    final_transform_vf = f.Execute(final_transform)
    final_transform_vf.SetDirection(reference_image.GetDirection())

    # output_filepath = write_vector_field(final_transform_vf, case=f'0{case}', output_folder=OUTPUT_FOLDER)

    fixed_image = np.swapaxes(fixed_image, 0, 2)
    fixed_image = sitk.GetImageFromArray(fixed_image)

    # disp_field = nib.load(output_filepath).get_fdata()
    #
    # tre_before = compute_tre(
    #     disp=None,
    #     fix_lms=fixed_landmarks,
    #     mov_lms=moving_landmarks,
    #     spacing_mov=(1.5,) * 3,
    # )
    # tre_after = compute_tre(
    #     disp=disp_field,
    #     fix_lms=fixed_landmarks,
    #     mov_lms=moving_landmarks,
    #     spacing_mov=(1.5,) * 3,
    # )

    vf = sitk.GetArrayFromImage(final_transform_vf)
    vf = np.swapaxes(vf, 0, 2)

    vf = torch.as_tensor(vf[np.newaxis], device="cuda:0")
    ml = torch.as_tensor(moving_landmarks[np.newaxis], device="cuda:0")
    fl = torch.as_tensor(fixed_landmarks[np.newaxis], device="cuda:0")

    tre_loss = TRELoss(image_spacing=(1.5, 1.5, 1.5), apply_sqrt=True).to("cuda:0")
    loss_before = float(tre_loss(vf * 0.0, ml, fl))
    loss_after = float(tre_loss(vf, ml, fl))

    tre_before = compute_tre_sitk(
        fix_lms=fixed_landmarks,
        mov_lms=moving_landmarks,
        spacing_mov=(1.5,) * 3,
    )
    tre_after = compute_tre_sitk(
        fix_lms=fixed_landmarks,
        mov_lms=moving_landmarks,
        transform=final_transform,
        ref_img=fixed_image,
        spacing_mov=(1.5,) * 3,
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
