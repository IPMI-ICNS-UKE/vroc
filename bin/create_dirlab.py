import re
from pathlib import Path

import SimpleITK as sitk
import numpy as np


def read_meta(filepath: Path) -> dict:
    with open(filepath, "rt") as f:
        lines = f.readlines()

    match = re.search("(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", lines[0])
    image_shape = tuple(int(val) for val in match.groups())

    match = re.search(
        "(([0-9]*[.])?[0-9]+)\s*x\s*(([0-9]*[.])?[0-9]+)\s*x\s*(([0-9]*[.])?[0-9]+)",
        lines[1],
    )
    voxel_spacing = tuple(float(val) for val in match.groups()[0::2])

    meta = dict(image_shape=image_shape, voxel_spacing=voxel_spacing)

    return meta


def read_image(filepath: Path) -> sitk.Image:
    folder = filepath.parents[1]
    meta = read_meta(folder / "meta.txt")

    image = np.fromfile(filepath, dtype=np.int16)
    image = image - 1024
    image = np.clip(image, -1024, 3071)
    image = image.reshape(meta["image_shape"][::-1])
    image = np.flip(image, axis=0)

    image = sitk.GetImageFromArray(image)
    image = sitk.DICOMOrient(image, 'LPI')
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetSpacing(meta["voxel_spacing"])

    return image


if __name__ == "__main__":
    DIRLAB_FOLDER = Path("/home/tsentker/data/dirlab2022/")

    for image_filepath in sorted(DIRLAB_FOLDER.glob("**/*.img")):

        phase = int(re.search("T(\d\d)", str(image_filepath)).group(1)) // 10

        print(phase, image_filepath)
        image = read_image(image_filepath)
        output_filepath = image_filepath.parent / f"phase_{phase}.mha"
        sitk.WriteImage(image, str(output_filepath))


