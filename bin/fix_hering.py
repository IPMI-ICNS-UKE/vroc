from pathlib import Path

import numpy as np
import SimpleITK as sitk

INPUT_FOLDER = Path("/datalake/learn2reg/NLST")
OUTPUT_FOLDER = Path("/datalake/learn2reg/NLST_fixed")

for image_filepath in INPUT_FOLDER.glob("**/*.nii.gz"):
    print(image_filepath)
    output_filepath = OUTPUT_FOLDER / image_filepath.relative_to(INPUT_FOLDER)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    image = sitk.ReadImage(str(image_filepath))

    image_array = sitk.GetArrayFromImage(image)
    image_array = image_array[:, ::-1, :]

    # this is a NOOP for masks
    image_array = np.clip(image_array, -1024, 3071)

    fixed_image = sitk.GetImageFromArray(image_array)
    fixed_image.CopyInformation(image)

    sitk.WriteImage(fixed_image, str(output_filepath))
