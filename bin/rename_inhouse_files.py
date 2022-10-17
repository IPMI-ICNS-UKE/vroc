from pathlib import Path
from typing import List

import yaml


def collect_file_paths(folders: List[Path]):
    file_names = {
        0: ("phase00*", "bin_00*", "*HERZPHASE -100%*", "T00-Case??_fixed*"),
        1: ("phase01*", "bin_01*", "*HERZPHASE -80%*", "T10-Case??_fixed*"),
        2: ("phase02*", "bin_02*", "*HERZPHASE -60%*", "T20-Case??_fixed*"),
        3: ("phase03*", "bin_03*", "*HERZPHASE -40%*", "T30-Case??_fixed*"),
        4: ("phase04*", "bin_04*", "*HERZPHASE -20%*", "T40-Case??_fixed*"),
        5: (
            "phase05*",
            "bin_05*",
            "*HERZPHASE 0%*",
            "*HERZPHASE -0%*",
            "T50-Case??_fixed*",
        ),
        6: ("phase06*", "bin_06*", "*HERZPHASE 20%*", "T60-Case??_fixed*"),
        7: ("phase07*", "bin_07*", "*HERZPHASE 40%*", "T70-Case??_fixed*"),
        8: ("phase08*", "bin_08*", "*HERZPHASE 60%*", "T80-Case??_fixed*"),
        9: ("phase09*", "bin_09*", "*HERZPHASE 80%*", "T90-Case??_fixed*"),
        "average": ("Resp  3.0  B31f  Average CT*", "pseudo_average*"),
    }

    collected = []
    for folder in folders:
        _collected = {}
        for file_name, patterns in file_names.items():
            for pattern in patterns:
                if matching_file_name := list(folder.rglob(pattern)):
                    _collected[file_name] = matching_file_name[0]
                    break

        if (meta_filepath := folder / "metadata.yml").exists():
            with open(meta_filepath, "r") as f:
                _collected["meta"] = yaml.safe_load(f)
        collected.append(_collected)

    return collected


if __name__ == "__main__":
    FOLDER = Path("/datalake_fast/4d_ct_lung_uke_artifact_free")

    patients = collect_file_paths(list(FOLDER.glob("*")))

    for patient in patients:
        for i_bin in range(10):
            filepath = patient[i_bin]
            filepath.rename(filepath.parent / f"phase_{i_bin:02d}.nii")

        patient["average"].rename(patient["average"].parent / f"average.nii")
