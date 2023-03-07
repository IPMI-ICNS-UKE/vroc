import logging
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import click
import numpy as np
import SimpleITK as sitk
import torch
import yaml

from vroc.helper import calculate_sha256_checksum, read_landmarks, write_landmarks
from vroc.logger import init_fancy_logging
from vroc.models import Unet3d
from vroc.segmentation import LungSegmenter3D

# the following information/data was taken/downloaded
# from the official website (as of UTC TIMESTAMP)
TIMESTAMP = 1677854057
DATASET_NAME = "DIR-Lab COPDgene"
WEBSITE = "https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/copdgene.html"
DOCUMENTATION = """Included with each case are 2 text files containing coordinate information for the reference landmarks on the inhalation (iBH-CT) and exhalation (eBH-CT) breath-hold CT images:

caseID_300_iBH_xyz_r1.txt, and
caseID_300_eBH_xyz_r1.txt"""


RAW_META = {
    1: """Image Dims: 512 x 512 x 121
Voxels (mm): 0.625 x 0.625 x 2.5
Features (#): 773
Displacement (mm): 25.90 (11.57)
Repeats (#/#): 150/3
Observers (mm): 0.65 (0.73)
Lowest Error (mm): 0.71 (0.81)""",
    2: """Image Dims: 512 x 512 x 102
Voxels (mm): 0.645 x 0.645 x 2.5
Features (#): 612
Displacement (mm): 21.77 (6.46)
Repeats (#/#): 150/3
Observers (mm): 1.06 (1.51)
Lowest Error (mm): 1.46 (2.28)""",
    3: """Image Dims: 512 x 512 x 126
Voxels (mm): 0.652 x 0.652 x 2.5
Features (#): 1172
Displacement (mm): 12.29 (6.39)
Repeats (#/#): 150/3
Observers (mm): 0.58 (0.87)
Lowest Error (mm): 0.77 (0.77)""",
    4: """Image Dims: 512 x 512 x 126
Voxels (mm): 0.590 x 0.590 x 2.5
Features (#): 786
Displacement (mm): 30.90 (13.49)
Repeats (#/#): 150/3
Observers (mm): 0.71 (0.96)
Lowest Error (mm): Observer Uncertainty Threshold""",
    5: """Image Dims: 512 x 512 x 131
Voxels (mm): 0.647 x 0.647 x 2.5
Features (#): 1029
Displacement (mm): 30.90 (14.05)
Repeats (#/#): 150/3
Observers (mm): 0.65 (0.87)
Lowest Error (mm): 0.71 (0.83)""",
    6: """Image Dims: 512 x 512 x 119
Voxels (mm): 0.633 x 0.633 x 2.5
Features (#): 633
Displacement (mm): 28.32 (9.20)
Repeats (#/#): 150/3
Observers (mm): 1.06 (2.38)
Lowest Error (mm): Observer Uncertainty Threshold""",
    7: """Image Dims: 512 x 512 x 112
Voxels (mm): 0.625 x 0.625 x 2.5
Features (#): 575
Displacement (mm): 21.66 (7.66)
Repeats (#/#): 150/3
Observers (mm): 0.65 (0.78)
Lowest Error (mm): 0.74 (1.06)""",
    8: """Image Dims: 512 x 512 x 115
Voxels (mm): 0.586 x 0.586 x 2.5
Features (#): 791
Displacement (mm): 25.57 (13.61)
Repeats (#/#): 150/3
Observers (mm): 0.96 (3.07)
Lowest Error (mm): Observer Uncertainty Threshold""",
    9: """Image Dims: 512 x 512 x 116
Voxels (mm): 0.664 x 0.664 x 2.5
Features (#): 447
Displacement (mm): 14.84 (10.01)
Repeats (#/#): 150/3
Observers (mm): 1.01 (2.54)
Lowest Error (mm): Observer Uncertainty Threshold""",
    10: """Image Dims: 512 x 512 x 135
Voxels (mm): 0.742 x 0.742 x 2.5
Features (#): 480
Displacement (mm): 22.48 (10.64)
Repeats (#/#): 150/3
Observers (mm): 0.87 (1.65)
Lowest Error (mm): Observer Uncertainty Threshold""",
}

# SHA256 hashes of downloaded files named copd1.zip, ..., copd10.zip
# dict key 1 corresponds to copd1.zip
FILE_CHECKSUMS = {
    1: "6f82be9534803d378c764683eb42028a48f6328859428c3c8978836722632370",
    2: "7c930297ebb77a30c7e3240846e6cbc22772e9bf17c04639e93f648d9558aab3",
    3: "faf3c28cfe4bb8d4b47ddf98a1452b6bc2d0b548455c2a5542685f5eedcd0cf3",
    4: "e7de222b6b9c67033e6403dfef78ba76b9b1e870c8ca33c1d3ae148cfae709f0",
    5: "954920fbbec649e5262cfa4807aec8f27cf4dbc255951f30c43c89590025c20a",
    6: "32cd099698622493b010a1c13f1395a73f42cc7ab130d93e5cb897b003098567",
    7: "d0fb693f86bf2bb61983a5e85eabb43ba7158a13d2944934c81a3a07c7d317dd",
    8: "13ea63623b148f4c7e9f8cc11688bcc2674b9caa50c784eae972cccccf4431f1",
    9: "4c45cf1a5e9d4b76cf03d305d2e255f397d4968449bf9cb56f942e7064d0d9a6",
    10: "985fd5d4fa8d8dab8837925ace19ad0f881750e1583acfcb70995f00efb2b262",
}


def read_meta(case_id: int) -> dict:

    decimal_regex = "([0-9]*[.])?[0-9]+"

    raw_meta = RAW_META[case_id]
    lines = raw_meta.split("\n")

    match = re.search("(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", lines[0])
    image_shape = [int(val) for val in match.groups()]

    match = re.search(
        f"(?P<x>{decimal_regex})\s*x\s*"
        f"(?P<y>{decimal_regex})\s*x\s*"
        f"(?P<z>{decimal_regex})",
        lines[1],
    )
    image_spacing = [float(match.groupdict()[axis]) for axis in ("x", "y", "z")]

    match = re.search(
        f"(?P<mean>{decimal_regex})\s\((?P<std>{decimal_regex})\)", lines[5]
    )
    observer_tre_mean = float(match.groupdict()["mean"])
    observer_tre_std = float(match.groupdict()["std"])

    meta = dict(
        image_shape=image_shape,
        image_spacing=image_spacing,
        observer_tre_mean=observer_tre_mean,
        observer_tre_std=observer_tre_std,
    )

    return meta


def read_image_data(filepath: Path, meta: dict):
    image = np.fromfile(filepath, dtype=np.int16)
    image = image - 1024
    image = np.clip(image, -1024, 3071)
    image = image.reshape(meta["image_shape"][::-1])
    image = np.flip(image, axis=0)

    image = sitk.GetImageFromArray(image)
    image.SetSpacing(meta["image_spacing"])

    return image


@click.command()
@click.argument(
    "input-folder",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--output-folder",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output folder for converted data set",
    show_default=True,
)
@click.option(
    "--lung-segmenter-weights",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Filepath of model weights for lung segmenter. "
        "If None, no lung masks will be created."
    ),
    show_default=True,
)
@click.option(
    "--gpu-device",
    type=str,
    default="cuda:0",
    help="CUDA device to use for lung segmenter",
    show_default=True,
)
def convert(
    input_folder: Path,
    output_folder: Path,
    lung_segmenter_weights: Path,
    gpu_device: str,
):
    if not output_folder:
        output_folder = input_folder.parent / "converted"
    output_folder.mkdir(exist_ok=True)

    logger.info("Start converting using the following config:")
    logger.info(f"{input_folder=!s}")
    logger.info(f"{output_folder=!s}")
    logger.info(f"{lung_segmenter_weights=!s}")
    logger.info(f"{gpu_device=}")

    if lung_segmenter_weights:
        model = Unet3d(n_levels=4, filter_base=32)
        state = torch.load(lung_segmenter_weights)
        model.load_state_dict(state["model"])
        lung_segmenter = LungSegmenter3D(model=model, device=gpu_device)
    else:
        lung_segmenter = None

    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for i_case in range(1, 11):
            logger.info(f"Extracting {DATASET_NAME} case {i_case}")

            file_checksum = calculate_sha256_checksum(
                input_folder / f"copd{i_case}.zip"
            )
            if file_checksum != FILE_CHECKSUMS[i_case]:
                logger.warning(
                    f"File checksum mismatch for {DATASET_NAME} case {i_case}"
                )
            else:
                logger.info(f"File checksum checked for {DATASET_NAME} case {i_case}")

            with ZipFile(input_folder / f"copd{i_case}.zip", "r") as f:
                f.extractall(tmp_dir)

            patient_input_folder = tmp_dir / f"copd{i_case}"

            # output folders
            patient_output_folder = output_folder / f"case_{i_case:02d}"
            image_output_folder = patient_output_folder / "images"
            masks_output_folder = patient_output_folder / "masks"
            landmarks_output_folder = patient_output_folder / "landmarks"
            keypoints_output_folder = patient_output_folder / "keypoints"
            patient_output_folder.mkdir(exist_ok=True)
            image_output_folder.mkdir(exist_ok=True)
            masks_output_folder.mkdir(exist_ok=True)
            landmarks_output_folder.mkdir(exist_ok=True)
            keypoints_output_folder.mkdir(exist_ok=True)

            landmarks = {
                0: read_landmarks(
                    patient_input_folder / f"copd{i_case}_300_iBH_xyz_r1.txt"
                ),
                5: read_landmarks(
                    patient_input_folder / f"copd{i_case}_300_eBH_xyz_r1.txt"
                ),
            }

            # write full (300) landmarks
            # landmarks are valid for both directions (5 to 0 and 0 to 5 registration)
            write_landmarks(
                landmarks[0], landmarks_output_folder / "moving_landmarks_00_to_05.csv"
            )
            write_landmarks(
                landmarks[5], landmarks_output_folder / "fixed_landmarks_00_to_05.csv"
            )

            write_landmarks(
                landmarks[0], landmarks_output_folder / "fixed_landmarks_05_to_00.csv"
            )
            write_landmarks(
                landmarks[5], landmarks_output_folder / "moving_landmarks_05_to_00.csv"
            )

            for i_phase in (0, 5):
                logger.info(f"Processing folder {patient_input_folder=!s}, {i_phase=}")

                in_or_ex = "i" if i_case == 0 else "e"

                meta = read_meta(i_case)
                image = read_image_data(
                    patient_input_folder / f"copd{i_case}_{in_or_ex}BHCT.img", meta=meta
                )

                lung_segmentation = lung_segmenter.segment(image, subsample=1.5)

                sitk.WriteImage(
                    image, str(image_output_folder / f"phase_{i_phase:02d}.nii")
                )
                sitk.WriteImage(
                    lung_segmentation,
                    str(masks_output_folder / f"lung_phase_{i_phase:02d}.nii.gz"),
                )
                with open(patient_output_folder / "metadata.yaml", "wt") as f_meta:
                    yaml.dump(meta, f_meta)


if __name__ == "__main__":
    init_fancy_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    convert()
