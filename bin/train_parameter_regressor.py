import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from vroc.feature_extractor import extract_histogram_features
from vroc.hyperopt_database.client import DatabaseClient


def build_param_matrix(runs: pd.DataFrame):
    runs = runs[["n_levels", "iterations", "tau", "sigma_x", "sigma_y", "sigma_z"]]
    runs = np.array(runs, dtype=np.float32)

    return runs


def build_matrix(runs: pd.DataFrame, features: dict, mode: str = "concat"):
    prepared_features = []
    for index, row in best_runs.iterrows():
        print(row.image)
        moving_image_name = f"{row.image}_0000"
        fixed_image_name = f"{row.image}_0001"

        moving_image = sitk.ReadImage(
            f"/datalake/learn2reg/NLST/imagesTr/{moving_image_name}.nii.gz"
        )
        moving_image = sitk.GetArrayFromImage(moving_image)
        moving_image = np.clip(moving_image, -1024, 3071)

        fixed_image = sitk.ReadImage(
            f"/datalake/learn2reg/NLST/imagesTr/{fixed_image_name}.nii.gz"
        )
        fixed_image = sitk.GetArrayFromImage(fixed_image)
        fixed_image = np.clip(fixed_image, -1024, 3071)

        moving_image_histogram = extract_histogram_features(moving_image)
        fixed_image_histogram = extract_histogram_features(fixed_image)
        histogram_difference = moving_image_histogram - fixed_image_histogram

        # merged_features = histogram_difference
        merged_features = np.hstack((moving_image_histogram, fixed_image_histogram))

        # moving_image_features = features[moving_image_name]
        # fixed_image_features = features[fixed_image_name]
        #
        # if mode == "concat":
        #     merged_features = np.concatenate(
        #         (moving_image_features, fixed_image_features), axis=1
        #     )
        # elif mode == 'difference':
        #     merged_features = moving_image_features - fixed_image_features
        # else:
        #     raise NotImplementedError

        prepared_features.append(
            {
                "uuid": index,
                # "moving_image_features": moving_image_features,
                # "fixed_image_features": fixed_image_features,
                "merged_features": merged_features,
            }
        )

    prepared_features = pd.DataFrame.from_records(prepared_features, index="uuid")
    matrix = runs.join(prepared_features)

    return matrix


if __name__ == "__main__":

    PARAM_NAMES = ["n_levels", "iterations", "tau", "sigma_x", "sigma_y", "sigma_z"]

    client = DatabaseClient("/datalake/learn2reg/merged_runs.sqlite")

    best_runs = client.fetch_best_runs(as_dataframe=True, k_best=1)

    with open("/datalake/learn2reg/nlst_features.pkl", "rb") as f:
        features = pickle.load(f)

    matrix = build_matrix(runs=best_runs, features=features, mode="concat")

    x = np.array([*matrix["merged_features"]]).squeeze()
    y = matrix[PARAM_NAMES]

    # subsample
    x = x[:, ::1]

    x, y = np.array(x), np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    model = XGBRegressor()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    r2 = r2_score(y_true=y_test, y_pred=y_pred)

    for i_param, param_name in enumerate(PARAM_NAMES):
        fig, ax = plt.subplots()
        fig.suptitle(param_name)
        ax.scatter(y_test[:, i_param], y_pred[:, i_param])
        ax.set_aspect("equal", "box")

        # grid = sns.jointplot(x=y_test[:, i_param], y=y_pred[:, i_param], kind="scatter", label=param_name)
