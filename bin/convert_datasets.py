import logging
from pathlib import Path

import pydicom as dcm
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


patient_folders = sorted(
    Path("/datalake/lung_ct_segmentation_challenge_2017/LCTSC").glob("LCTSC*")
)
patient_folders += sorted(
    Path("/datalake/nsclc_radiomics/NSCLC-Radiomics").glob("LUNG*")
)
patient_folders += sorted(Path("/datalake/rider_lung_ct/RIDER Lung CT").glob("RIDER*"))

for patient in sorted(patient_folders):
    print(patient.name)
    folders = get_dicom_folders(patient)

    ct_folders = [
        path for path, details in folders.items() if details["modality"] == "CT"
    ]
    rtstruct_folders = [
        path for path, details in folders.items() if details["modality"] == "RTSTRUCT"
    ]

    reference_ct = ct_folders[0]

    for ct_folder in ct_folders:
        convert_dicom(
            image_dicom_folder=ct_folder,
            image_output_filepath=ct_folder / "image.mha",
        )

    for rtstruct_folder in rtstruct_folders:
        convert_dicom(
            image_dicom_folder=reference_ct,
            rtstruct_folder=rtstruct_folder,
            segmentation_output_folder=rtstruct_folder / "segmentations",
        )
