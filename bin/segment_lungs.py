import logging
import re
from pathlib import Path
from typing import List

import click
import numpy as np
import SimpleITK as sitk

from vroc.common_types import PathLike
from vroc.decorators import convert
from vroc.helper import remove_suffixes
from vroc.segmentation import LungSegmenter2D


@convert("path", converter=Path)
def iterate_images(path: PathLike, skip_patterns: List[str] = None):
    path: Path
    logging.debug(f"Handle {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    if skip_patterns:
        for skip_pattern in skip_patterns:
            if re.search(skip_pattern, path.name):
                logging.warning(
                    f'Skipping "{path}" due to skip pattern "{skip_pattern}"'
                )
                return

    # single file
    if path.is_file():
        try:
            image = sitk.ReadImage(str(path))
            logging.info(f'Loaded image file "{path}"')
            yield {"filepath": path, "image": image}

        except RuntimeError:
            logging.warning(f"Cannot load {path} (probably not an image file)")

    elif path.is_dir():
        if not any(path.iterdir()):
            # folder is empty
            logging.warning(f"Folder {path} is empty")
        else:
            logging.debug(
                f"Folder {path} does not contain valid image files. Going deeper"
            )
            for child_path in sorted(path.glob("*")):
                yield from iterate_images(child_path, skip_patterns=skip_patterns)

    else:
        raise RuntimeError(f'Cannot handle "{path}"')


if __name__ == "__main__":
    from vroc.models import UNet

    lung_segmenter = LungSegmenter2D(
        model=UNet().to("cuda"),
        iter_axis=2,
        state_filepath=Path("/datalake/learn2reg/weights/lung_segmenter.pth"),
    )

    for data in iterate_images(
        "/datalake/learn2reg/copd_dirlab2022/data/copd01/Images/phase_e.mha",
        skip_patterns=["mask"],
    ):
        image = sitk.ReadImage(str(data["filepath"]))
        img_arr = sitk.GetArrayFromImage(image)
        img_arr = np.swapaxes(img_arr, 0, 2)
        lung_mask = lung_segmenter.segment(img_arr)
        lung_mask = np.swapaxes(lung_mask, 0, 2)
        lung_mask = lung_mask.astype(np.uint8)
        lung_mask = sitk.GetImageFromArray(lung_mask)
        lung_mask.CopyInformation(image)

        image_filename = remove_suffixes(data["filepath"]).name
        out_filepath = data["filepath"].parent / f"{image_filename}_lung_mask.nii.gz"
        sitk.WriteImage(lung_mask, str(out_filepath))
        print(out_filepath)
        break
