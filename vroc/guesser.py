from __future__ import annotations

from typing import Sequence

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from vroc.common_types import PathLike
from vroc.database.client import DatabaseClient
from vroc.logger import LoggerMixin


class ParameterGuesser(LoggerMixin):
    def __init__(
        self,
        database_filepath: PathLike,
        n_dimensions: int = 2,
        parameters_to_guess: Sequence[str] | None = None,
    ):
        self._client = DatabaseClient(database_filepath)
        self.n_dimensions = n_dimensions
        self._mapper = None
        self._nearest_neighbors = None
        self._embedded = None
        self._image_pairs = None
        self._parameters_to_guess = parameters_to_guess

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

        parameters = self._client.fetch_best_parameters(
            moving_image=nearest_image_pair["moving_image"],
            fixed_image=nearest_image_pair["fixed_image"],
        )

        if self._parameters_to_guess is not None:
            parameters = {
                param_name: param_value
                for (param_name, param_value) in parameters.items()
                if param_name in self._parameters_to_guess
            }

        return parameters
