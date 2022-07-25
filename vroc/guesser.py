import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from vroc.common_types import PathLike
from vroc.database.client import DatabaseClient
from vroc.logger import LoggerMixin


class ParameterGuesser(LoggerMixin):
    def __init__(self, database_filepath: PathLike, n_dimensions: int = 2):
        self._client = DatabaseClient(database_filepath)
        self.n_dimensions = n_dimensions
        self._mapper = None
        self._nearest_neighbors = None
        self._embedded = None
        self._image_pairs = None

    def fit(self):
        self._image_pairs = self._client.fetch_image_pair_features()

        features = np.array(
            [image_pair["features"] for image_pair in self._image_pairs]
        )

        features = features.reshape(len(features), -1)

        self.logger.info(f"Fitting UMAP on features with shape {features.shape}")

        self._mapper = umap.UMAP(
            n_neighbors=self.n_dimensions,
            min_dist=0.0,
            metric="euclidean",
            random_state=1337,
            init="random",
        )

        self._embedded = self._mapper.fit_transform(features)

        self._nearest_neighbors = NearestNeighbors(n_neighbors=1)
        self._nearest_neighbors.fit(self._embedded)

    def guess(self, features: np.ndarray) -> dict:
        if not self._mapper:
            raise RuntimeError("Please fit ParameterGuesser first")

        embedded = self._mapper.transform(features.reshape(1, -1))
        distances, indices = self._nearest_neighbors.kneighbors(embedded)

        index = int(indices.squeeze())
        nearest_image_pair = self._image_pairs[index]

        return {
            "iterations": 1000,
            "tau": 2.0,
            "sigma_x": 1.25,
            "sigma_y": 1.25,
            "sigma_z": 1.25,
            "n_levels": 3,
        }

        return self._client.fetch_best_parameters(
            moving_image=nearest_image_pair["moving_image"],
            fixed_image=nearest_image_pair["fixed_image"],
        )
