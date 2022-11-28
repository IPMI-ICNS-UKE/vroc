import logging
from pathlib import Path

import numpy as np
import pandas as pd
import umap

from vroc.hyperopt_database.client import DatabaseClient
from vroc.logger import LogFormatter
from vroc.plot import plot_embedding

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(LogFormatter())
logging.basicConfig(handlers=[handler])

logging.getLogger(__name__).setLevel(logging.DEBUG)
logging.getLogger("vroc").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

ROOT_DIR = Path("/datalake/learn2reg/NLST")

client = DatabaseClient("/datalake/learn2reg/param_sampling.sqlite")

image_pairs = client.fetch_image_pairs()

ohs = []
forces = []
hover_data = []
downscaling = 4
histo_plots = []
clim_max = 0

for image_pair in image_pairs:
    moving_image = image_pair["moving_image"]
    fixed_image = image_pair["fixed_image"]

    feature = client.fetch_image_pair_feature(
        moving_image=moving_image, fixed_image=fixed_image, feature_name="OH_16"
    )

    hover_data.append(
        {
            "image_pair": f"{moving_image.name} / {fixed_image.name}",
            "color": "orangered",
        }
    )

    ohs.append(feature)

ohs = np.array(ohs)
hover_data = pd.DataFrame.from_records(hover_data)

mapper = umap.UMAP(n_neighbors=2, min_dist=0.0, metric="euclidean", densmap=False)
mapper.fit(ohs.reshape(len(ohs), -1))
plot_embedding(
    mapper.embedding_,
    tooltip_images=ohs,
    patients=hover_data["image_pair"],
    colors=hover_data["color"],
)
