import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import umap
from dataset import NLSTDataset
from preprocessing import crop_background, resample_image_size
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from vroc.helper import rescale_range


def load_file(filepath: os.path) -> dict:
    assert os.path.isfile(filepath)
    data = pickle.load(open(filepath, "rb"))
    filenames = np.asarray(list(data.keys()))
    prepro_data = np.concatenate(list(data.values()), axis=0)
    return filenames, prepro_data


def draw_umap(
    image_features, n_neighbors=10, min_dist=0.1, n_components=2, metric="euclidean"
):
    """Finds a 2-dimensional embedding of image_features that approximates an
    underlying manifold and plots the results."""
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
    )
    u = fit.fit_transform(image_features)
    fig = plt.figure()

    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], u[:, 1])
    else:
        raise ValueError

    fig.savefig("umap.png")
    return u


def get_neighbors(
    x,
    keys,
    plot_cases,
    saving_dir,
    luna_dir="/media/lwimmert/5E57-FB01/learn2reg/luna16/images",
    nlst_dir="/media/lwimmert/5E57-FB01/learn2reg/NLST/imagesTr",
):
    """Based on umap-features calculates 5 nearest neighbors and returns
    cohorts (including ct image filenames).

    if plot_cases is true: plots all 5 ct images as one plot in
    saving_dir
    """
    nbrs = NearestNeighbors(
        n_neighbors=5, metric="euclidean", algorithm="ball_tree"
    ).fit(x)
    distances, indices = nbrs.kneighbors(x)
    summed_distances = distances.sum(axis=1).reshape(-1, 1)

    # for each
    cohorts = keys[indices]

    if plot_cases:
        assert os.path.isdir(saving_dir)
        for j, cohort in enumerate(tqdm(indices)):
            cases = keys[cohort]

            fig, ax = plt.subplots(5, 1)

            for i, img_filepath in enumerate(cases):
                if img_filepath.endswith(".mhd"):
                    dir_ = os.path.join(luna_dir)
                elif img_filepath.endswith(".nii.gz"):
                    dir_ = os.path.join(nlst_dir)
                else:
                    raise ValueError
                # preprocessing
                image_path = os.path.join(dir_, img_filepath)
                image = NLSTDataset.load_and_preprocess(image_path)
                image = crop_background(image)
                image = resample_image_size(image, new_size=(128, 128, 128))
                image = sitk.GetArrayFromImage(image)
                image = rescale_range(
                    image, input_range=(-1024, 3071), output_range=(0, 1)
                )

                # show arbitrary slice
                ax[i].imshow(image[:, 60, :])
            fig.savefig(
                os.path.join(
                    saving_dir,
                    f"summed_cohort_distance_{np.round(summed_distances[j].item(),3)}.png",
                )
            )
    return cohorts


def main(
    filepath_features: os.path = os.path.join(".", "features_data", "features.p"),
    plot_cases: bool = False,
    saving_dir: str = False,
) -> np.array:

    filenames, prepro_data = load_file(filepath_features)

    x = draw_umap(image_features=prepro_data)

    cohorts = get_neighbors(
        x, keys=filenames, plot_cases=plot_cases, saving_dir=saving_dir
    )
    return cohorts


if __name__ == "__main__":
    cohorts = main(
        plot_cases=True,
        saving_dir=os.path.join(".", "features_data", "neighbors_examples"),
    )
