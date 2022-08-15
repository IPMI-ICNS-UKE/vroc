from __future__ import annotations

import logging
import pickle
from collections import deque
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from vroc.blocks import SpatialTransformer
from vroc.dataset import NLSTDataset
from vroc.helper import dict_collate, rescale_range
from vroc.logger import LogFormatter
from vroc.loss import TRELoss, WarpedMSELoss
from vroc.models import DemonsVectorFieldBooster, VectorFieldBooster

logger = logging.getLogger(__name__)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(LogFormatter())
logging.basicConfig(handlers=[handler])

logging.getLogger(__name__).setLevel(logging.INFO)
logging.getLogger("vroc").setLevel(logging.INFO)

DEVICE = "cuda:0"

train_dataset = NLSTDataset(
    "/datalake/learn2reg/NLST", i_worker=None, n_worker=None, dilate_masks=1
)


model = DemonsVectorFieldBooster(shape=(224, 192, 224), n_iterations=5).to(DEVICE)
optimizer = Adam(model.parameters())
tre_loss = TRELoss(apply_sqrt=True)
mse_loss = WarpedMSELoss(shape=(224, 192, 224)).to(DEVICE)
optimizer.zero_grad()
i = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    collate_fn=partial(dict_collate, noop_keys=("precomputed_vector_fields")),
    shuffle=True,
)


accumulation_steps = 8

trailing_loss = deque(maxlen=100)

spatial_transformer = SpatialTransformer(shape=(224, 192, 224)).to(DEVICE)

for i_epoch in range(1000):
    for data in train_loader:

        moving_image = rescale_range(
            data["moving_image"], input_range=(-1024, 3071), output_range=(0, 1)
        )
        fixed_image = rescale_range(
            data["fixed_image"], input_range=(-1024, 3071), output_range=(0, 1)
        )

        moving_image = torch.as_tensor(moving_image, device=DEVICE)
        fixed_image = torch.as_tensor(fixed_image, device=DEVICE)
        # moving_mask = torch.as_tensor(data["moving_mask"], device=DEVICE)
        fixed_mask = torch.as_tensor(
            data["fixed_mask"], device=DEVICE, dtype=torch.bool
        )
        moving_keypoints = torch.as_tensor(data["moving_keypoints"], device=DEVICE)
        fixed_keypoints = torch.as_tensor(data["fixed_keypoints"], device=DEVICE)
        image_spacing = torch.as_tensor(data["image_spacing"][0], device=DEVICE)

        for filepath in data["precomputed_vector_fields"][0]:
            with open(filepath, "rb") as f:
                precomputed = pickle.load(f)

            affine_vector_field = torch.as_tensor(
                precomputed["affine_vector_field"][None], device=DEVICE
            )
            affine_moving_image = spatial_transformer(moving_image, affine_vector_field)

            varreg_vector_field = torch.as_tensor(
                precomputed["varreg_vector_field"][None], device=DEVICE
            )

            tre_before = precomputed["tre_before"].mean()
            tre_after_affine = precomputed["tre_after_affine"].mean()
            tre_after_varreg = precomputed["tre_after_varreg"].mean()

            # boost the vector field
            with torch.cuda.amp.autocast():
                composed_varreg_vector_field = (
                    varreg_vector_field
                    + spatial_transformer(affine_vector_field, varreg_vector_field)
                )

                warped_image_varreg = spatial_transformer(
                    moving_image, composed_varreg_vector_field
                )
                vector_field_boost = model(
                    warped_image_varreg, fixed_image, image_spacing
                )

                composed_boosted_vector_field = (
                    vector_field_boost
                    + spatial_transformer(
                        composed_varreg_vector_field, vector_field_boost
                    )
                )

                # warped_image_varreg = spatial_transformer(moving_image, composed_varreg_vector_field)
                # warped_image_boosted = spatial_transformer(moving_image,
                #                                    composed_boosted_vector_field)
                #
                # fixed_image = fixed_image.detach().cpu().numpy()
                # warped_image_varreg = warped_image_varreg.detach().cpu().numpy()
                # warped_image_boosted = warped_image_boosted.detach().cpu().numpy()
                # image_diff = warped_image_varreg - warped_image_boosted
                # varreg_vector_field = varreg_vector_field.detach().cpu().numpy()
                # boosted_vector_field = boosted_vector_field.detach().cpu().numpy()
                #
                # diff = varreg_vector_field - boosted_vector_field
                # fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
                # ax[0, 0].imshow(warped_image_varreg[0, 0, :, 95, :])
                # ax[0, 1].imshow(warped_image_boosted[0, 0, :, 95, :])
                # ax[0, 2].imshow(fixed_image[0, 0, :, 95, :])
                # ax[1, 0].imshow(varreg_vector_field[0, 2, :, 95, :], cmap='seismic', clim=(-20, 20))
                # ax[1, 1].imshow(boosted_vector_field[0, 2, :, 95, :],
                #              cmap='seismic', clim=(-20, 20))
                # ax[1, 2].imshow(diff[0, 2, :, 95, :],
                #              cmap='seismic', clim=(-0.5, 0.5))

                tre_after_boosting = tre_loss(
                    composed_boosted_vector_field,
                    moving_keypoints,
                    fixed_keypoints,
                    image_spacing,
                )

                mse_after_varreg = mse_loss(
                    moving_image, composed_varreg_vector_field, fixed_image, fixed_mask
                )
                mse_after_boosting = mse_loss(
                    moving_image, composed_boosted_vector_field, fixed_image, fixed_mask
                )

                tre_impovement = (
                    1
                    + (tre_after_boosting - tre_after_varreg)
                    / (tre_after_varreg + 1e-6)
                ) ** 2
                mse_improvement = (
                    1
                    + (mse_after_boosting - mse_after_varreg)
                    / (mse_after_varreg + 1e-6)
                ) ** 2

                loss = tre_impovement  # (tre_impovement + mse_improvement) / 2.0

            trailing_loss.append(float(loss))
            logger.info(
                f"[{i_epoch}] {filepath.name} TRE: "
                f"{tre_before:.2f} / "
                f"{tre_after_affine:.2f} / "
                f"{tre_after_varreg:.2f} / "
                f"{tre_after_boosting:.2f}, "
                f"Loss: {loss:.6f} (tre: {tre_impovement:.6f}, mse: {mse_improvement:.6f})"
                f"trailing {np.mean(trailing_loss):.6f}"
            )
            loss = loss / accumulation_steps
            loss.backward()
            i += 1

            if i % accumulation_steps == 0:
                logger.info("Perform optimizer step")
                optimizer.step()
                optimizer.zero_grad()
