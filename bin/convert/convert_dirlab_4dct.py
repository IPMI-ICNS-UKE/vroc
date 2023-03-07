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
DATASET_NAME = "DIR-Lab 4DCT"
WEBSITE = "https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html"
DOCUMENTATION = """For the 4DCT images, the full point set is identified between the maximum inhalation and exhalation component phase images. Additionally, a subset of 75 features has been propagated onto each of the expiratory phase images (i.e., T00, T10, T20, T30, T40, and T50).

Each case is identified according to the given label.

The image dimensions are given in voxel units, and the voxel dimensions are given in millimeters.
The "# Features" designates the total quantity of unique landmark features identified for each case.
The "Displacement" shows the mean (and standard deviation) displacement of the complete primary feature set.
The entries under "# Repeats" are formatted as (Nm< / Nobs), where Nm is the number of repeat registration measurements performed by each of Nobs independent observers.
The "Observers" column shows the combined mean (and standard deviation) repeat registration error for the set of Nobs data sets."""


RAW_META = {
    1: """Image Dims: 256 x 256 x 94
Voxels (mm): 0.97 x 0.97 x 2.5
Features (#): 1280
Displacement (mm): 4.01 (2.91)
Repeats (#/#): 200/3
Observers (mm): 0.85 (1.24)
Lowest Error (mm): Observer Uncertainty Threshold""",
    2: """Image Dims: 256 x 256 x 112
Voxels (mm): 1.16 x 1.16 x 2.5
Features (#): 1487
Displacement (mm): 4.65 (4.09)
Repeats (#/#): 200/3
Observers (mm): 0.70 (0.99)
Lowest Error (mm): 0.72 (0.87)""",
    3: """Image Dims: 256 x 256 x 104
Voxels (mm): 1.15 x 1.15 x 2.5
Features (#): 1561
Displacement (mm): 6.73 (4.21)
Repeats (#/#): 200/3
Observers (mm): 0.77 (1.01)
Lowest Error (mm): 0.90 (1.05)""",
    4: """Image Dims: 256 x 256 x 99
Voxels (mm): 1.13 x 1.13 x 2.5
Features (#): 1166
Displacement (mm): 9.42 (4.81)
Repeats (#/#): 200/3
Observers (mm): 1.13 (1.27)
Lowest Error (mm): 1.21 (1.19)""",
    5: """Image Dims: 256 x 256 x 106
Voxels (mm): 1.10 x 1.10 x 2.5
Features (#): 1268
Displacement (mm): 7.10 (5.14)
Repeats (#/#): 200/3
Observers (mm): 0.92 (1.16)
Lowest Error (mm): 1.07 (1.46)""",
    6: """Image Dims: 512 x 512 x 128
Voxels (mm): 0.97 x 0.97 x 2.5
Features (#): 419
Displacement (mm): 11.10 (6.98)
Repeats (#/#): 150/3
Observers (mm): 0.97 (1.38)
Lowest Error (mm): Observer Uncertainty Threshold""",
    7: """Image Dims: 512 x 512 x 136
Voxels (mm): 0.97 x 0.97 x 2.5
Features (#): 398
Displacement (mm): 11.59 (7.87)
Repeats (#/#): 150/3
Observers (mm): 0.81 (1.32)
Lowest Error (mm): Observer Uncertainty Threshold""",
    8: """Image Dims: 512 x 512 x 128
Voxels (mm): 0.97 x 0.97 x 2.5
Features (#): 476
Displacement (mm): 15.16 (9.11)
Repeats (#/#): 150/3
Observers (mm): 1.03 (2.19)
Lowest Error (mm): Observer Uncertainty Threshold""",
    9: """Image Dims: 512 x 512 x 128
Voxels (mm): 0.97 x 0.97 x 2.5
Features (#): 342
Displacement (mm): 7.82 (3.99)
Repeats (#/#): 150/3
Observers (mm): 0.75 (1.09)
Lowest Error (mm): 0.91 (0.93)""",
    10: """Image Dims: 512 x 512 x 120
Voxels (mm): 0.97 x 0.97 x 2.5
Features (#): 435
Displacement (mm): 7.63 (6.54)
Repeats (#/#): 150/3
Observers (mm): 0.86 (1.45)
Lowest Error (mm): Observer Uncertainty Threshold""",
}

# SHA256 hashes of downloaded files named Case1Pack.zip, ..., Case10Pack.zip
# dict key 1 corresponds to Case1Pack.zip
FILE_CHECKSUMS = {
    1: "5589732668c3694e46a3c0eda4ccaceb4e634de08c633bcfb0362644caf1287a",
    2: "237b9cf423967f86149c3680029b7c6afae5c3e9b315c91a00d43953362717c8",
    3: "824d9cb93891a5fd73835aae28cd3b21cdb22c81f552f5fb958a06c63b6c76fc",
    4: "cd5603cb2611911f04b188dd1f26f988d6740fe077dba02734b7f50b344bda0e",
    5: "6bf6afffcc01b0039f84c4f4ab4dd9aae9807d35b217196430697865a8dbddc2",
    6: "a48a87ff5feee9decd8f7e27e2053de904977406e1f74e21b1247de4c40de916",
    7: "8154f7f84ad927a135df11e6c10d5db8d2f7a92d346cf17978ed8069a569389a",
    8: "29b5f777a5798e72a1ec1a089e0f3d4ee690bcdc4b62a78b8f83882037a35743",
    9: "138282e9ce4ab55093611456776a7ef7baa042d2a53a8da0d37f2579a1611a57",
    10: "c402b0d3f99a1fe470e758816a3f4d1f07248ac8f34bc0cf83cea239c6e410f0",
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
                input_folder / f"Case{i_case}Pack.zip"
            )
            if file_checksum != FILE_CHECKSUMS[i_case]:
                logger.warning(
                    f"File checksum mismatch for {DATASET_NAME} case {i_case}"
                )
            else:
                logger.info(f"File checksum checked for {DATASET_NAME} case {i_case}")

            with ZipFile(input_folder / f"Case{i_case}Pack.zip", "r") as f:
                f.extractall(tmp_dir)

            # dirlab folders and files are not named consistently
            if i_case == 8:
                patient_input_folder = tmp_dir / f"Case{i_case}Deploy"
            else:
                patient_input_folder = tmp_dir / f"Case{i_case}Pack"

            if i_case <= 5:
                landmarks_input_folder = patient_input_folder / "ExtremePhases"
                landmarks_base_filename = f"Case{i_case}_300"
            else:
                landmarks_input_folder = patient_input_folder / "extremePhases"
                landmarks_base_filename = f"case{i_case}_dirLab300"

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
                    landmarks_input_folder / f"{landmarks_base_filename}_T00_xyz.txt"
                ),
                1: read_landmarks(
                    patient_input_folder / f"Sampled4D/case{i_case}_4D-75_T10.txt"
                ),
                2: read_landmarks(
                    patient_input_folder / f"Sampled4D/case{i_case}_4D-75_T20.txt"
                ),
                3: read_landmarks(
                    patient_input_folder / f"Sampled4D/case{i_case}_4D-75_T30.txt"
                ),
                4: read_landmarks(
                    patient_input_folder / f"Sampled4D/case{i_case}_4D-75_T40.txt"
                ),
                5: read_landmarks(
                    landmarks_input_folder / f"{landmarks_base_filename}_T50_xyz.txt"
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

            # write subset (75) landmarks
            for i_moving in range(2, 5):
                for i_fixed in range(2, 5):
                    if i_moving == i_fixed:
                        continue
                    write_landmarks(
                        landmarks[i_moving],
                        landmarks_output_folder
                        / f"moving_landmarks_{i_moving:02d}_to_{i_fixed:02d}.csv",
                    )
                    write_landmarks(
                        landmarks[i_fixed],
                        landmarks_output_folder
                        / f"fixed_landmarks_{i_moving:02d}_to_{i_fixed:02d}.csv",
                    )

            for i_phase in range(10):
                logger.info(f"Processing folder {patient_input_folder=!s}, {i_phase=}")

                # dirlab images are not named consistently, thus trial and error
                possible_filenames = (
                    f"case{i_case}_T{i_phase * 10:02d}.img",
                    f"case{i_case}_T{i_phase * 10:02d}_s.img",
                    f"case{i_case}_T{i_phase * 10:02d}-ssm.img",
                )
                for filename in possible_filenames:
                    if (
                        image_filepath := patient_input_folder / "Images" / filename
                    ).exists():
                        break
                else:
                    raise FileNotFoundError(
                        f"No matching file found for {patient_input_folder=!s}, {i_phase=}"
                    )

                meta = read_meta(i_case)
                image = read_image_data(image_filepath, meta=meta)

                sitk.WriteImage(
                    image, str(image_output_folder / f"phase_{i_phase:02d}.nii")
                )

                if lung_segmenter:
                    logger.info("Create lung segmentation")
                    lung_segmentation = lung_segmenter.segment(image, subsample=1.5)
                    sitk.WriteImage(
                        lung_segmentation,
                        str(masks_output_folder / f"lung_phase_{i_phase:02d}.nii.gz"),
                    )
                else:
                    logger.info("Skipping creation of lung segmentation")

                with open(patient_output_folder / "metadata.yaml", "wt") as f_meta:
                    yaml.dump(meta, f_meta)


if __name__ == "__main__":
    init_fancy_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    convert()
