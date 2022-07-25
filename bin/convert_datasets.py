import logging
import re
from functools import reduce
from pathlib import Path

import pydicom as dcm
import SimpleITK as sitk
from ipmi.common.dataio.radiotherapy import convert_dicom

logging.basicConfig(level=logging.INFO)


def get_dicom_folders(folder: Path):
    folders = {}
    for dcm_file in folder.rglob("*.dcm"):
        folder = dcm_file.parent
        modality = dcm.read_file(dcm_file).Modality
        try:
            folders[folder]["files"].append(dcm_file)
            if not folders[folder]["modality"] == modality:
                raise RuntimeError("Multipe modalities in one folder")

        except KeyError:
            folders[folder] = {}
            folders[folder]["files"] = [dcm_file]
            folders[folder]["modality"] = modality

    return folders


dicom_datasets = [
    {
        "patient_folders": sorted(
            Path("/datalake/lung_ct_segmentation_challenge_2017/LCTSC").glob("LCTSC*")
        ),
        "lung_segmentations": [re.compile(r"lung.*")],
    },
    {
        "patient_folders": sorted(
            Path("/datalake/nsclc_radiomics/NSCLC-Radiomics").glob("LUNG*")
        ),
        "lung_segmentations": [re.compile(r"lung.*"), re.compile(r"gtv.*")],
    },
    {
        "patient_folders": sorted(
            Path("/datalake/rider_lung_ct/RIDER Lung CT").glob("RIDER*")
        ),
        "lung_segmentations": [],
    },
]

datasets = [
    {
        "images": sorted(Path("/datalake/mega/luna16/images").glob("*.mhd")),
        "lung_segmentations": sorted(
            Path("/datalake/mega/luna16/segmentations").glob("*.mhd")
        ),
        "lung_labels": (3, 4),
    },
]

for dataset in dicom_datasets:
    for patient in dataset["patient_folders"]:
        print(patient.name)
        folders = get_dicom_folders(patient)

        ct_folders = [
            path for path, details in folders.items() if details["modality"] == "CT"
        ]
        rtstruct_folders = [
            path
            for path, details in folders.items()
            if details["modality"] == "RTSTRUCT"
        ]

        reference_ct = ct_folders[0]

        for ct_folder in ct_folders:
            if not (ct_folder / "image.mha").exists():
                convert_dicom(
                    image_dicom_folder=ct_folder,
                    image_output_filepath=ct_folder / "image.mha",
                )

        for rtstruct_folder in rtstruct_folders:
            if not (rtstruct_folder / "segmentations").exists():
                convert_dicom(
                    image_dicom_folder=reference_ct,
                    rtstruct_folder=rtstruct_folder,
                    segmentation_output_folder=rtstruct_folder / "segmentations",
                )

            if not (rtstruct_folder / "segmentations" / "lungs.nii.gz").exists():
                # merge segmentations, e.g. Lung_Left.nii.gz + Lung_Right.nii.gz
                to_merge = []
                for filepath in (rtstruct_folder / "segmentations").glob("*nii.gz"):

                    for pattern in dataset["lung_segmentations"]:
                        if re.search(pattern, filepath.name.lower()):
                            print("Merge", filepath.name)
                            segmentation = sitk.ReadImage(str(filepath), sitk.sitkUInt8)
                            to_merge.append(segmentation)
                if to_merge:
                    lung_segmentation = reduce(lambda a, b: a | b, to_merge)
                    # fill holes
                    lung_segmentation = sitk.BinaryFillhole(lung_segmentation)
                    sitk.WriteImage(
                        lung_segmentation,
                        str(rtstruct_folder / "segmentations" / "lungs.nii.gz"),
                    )


for dataset in datasets:
    for image_filepath, segmentation_filepath in zip(
        dataset["images"], dataset["lung_segmentations"]
    ):
        print(image_filepath.name)
        lung_segmentation_filepath = (
            segmentation_filepath.parent
            / "lung_segmentations"
            / segmentation_filepath.name
        )
        lung_segmentation_filepath.parent.mkdir(exist_ok=True)

        segmentation = sitk.ReadImage(str(segmentation_filepath), sitk.sitkUInt8)
        lung_segmentation = sitk.Image(segmentation.GetSize(), sitk.sitkUInt8)
        lung_segmentation.CopyInformation(segmentation)

        for label in dataset["lung_labels"]:
            lung_segmentation = lung_segmentation | (segmentation == label)

        sitk.WriteImage(
            lung_segmentation, str(lung_segmentation_filepath.with_suffix(".nii.gz"))
        )
