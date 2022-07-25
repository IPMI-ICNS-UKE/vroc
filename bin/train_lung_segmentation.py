from functools import partial, reduce
from operator import add
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from aim import Figure, Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from vroc.dataset import LungCTSegmentationDataset
from vroc.helper import dict_collate
from vroc.logger import LogFormatter
from vroc.metrics import dice_coefficient
from vroc.models import LungCTSegmentationUnet3d
from vroc.trainer import BaseTrainer, MetricType

datasets = {
    # "lctsc": {
    #     "images": sorted(
    #         Path("/datalake_fast/lung_segmentation_datasets/lung_ct_segmentation_challenge_2017/LCTSC").glob(
    #             "**/image.mha"
    #         )
    #     ),
    #     "masks": sorted(
    #         Path("/datalake_fast/lung_segmentation_datasets/lung_ct_segmentation_challenge_2017/LCTSC").glob(
    #             "**/lungs.nii.gz"
    #         )
    #     ),
    # },
    # "nsclc": {
    #     "images": sorted(
    #         Path("/datalake/nsclc_radiomics/NSCLC-Radiomics").glob("**/image.mha")
    #     ),
    #     "masks": sorted(
    #         Path("/datalake/nsclc_radiomics/NSCLC-Radiomics").glob("**/lungs.nii.gz")
    #     ),
    # },
    "luna": {
        "images": sorted(
            Path("/datalake_fast/lung_segmentation_datasets/luna16/images").glob(
                "*.mhd"
            )
        ),
        "masks": sorted(
            Path("/datalake_fast/lung_segmentation_datasets/luna16/segmentations").glob(
                "*.mhd"
            )
        ),
        "labels": (3, 4),
    },
    # "structseg": {
    #     "images": sorted(
    #         Path("/datalake_fast/lung_segmentation_datasets/task3_thoracic_oar/").glob("**/data.nii.gz")
    #     ),
    #     "masks": sorted(
    #         Path("/datalake_fast/lung_segmentation_datasets/task3_thoracic_oar/").glob(
    #             "**/label.nii.gz"
    #         )
    #     ),
    #     "labels": (1, 2),
    # },
    # "learn2reg_nlst": {
    #     "images": sorted(
    #         Path("/datalake_fast/lung_segmentation_datasets/NLST_fixed/imagesTr").glob("*.nii.gz")
    #     ),
    #     "masks": sorted(
    #         Path("/datalake_fast/lung_segmentation_datasets/NLST_fixed/masksTr").glob("*.nii.gz")
    #     ),
    # },
}


all_images = reduce(add, [d["images"] for d in datasets.values()])
all_masks = reduce(add, [d["masks"] for d in datasets.values()])
all_labels = reduce(
    add, [[d.get("labels")] * len(d["masks"]) for d in datasets.values()]
)


class LungCTSegmentationTrainer(BaseTrainer):
    METRICS = {
        "loss": MetricType.SMALLER_IS_BETTER,
        "dice": MetricType.LARGER_IS_BETTER,
    }

    def train_on_batch(self, data: dict) -> dict:

        images = data["image"].to(self.device)
        masks = data["mask"].to(self.device)

        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = self.model(images)
            losses = self.loss_function(outputs, masks)
            loss_per_image = losses.mean(dim=(1, 2, 3, 4))
            loss = loss_per_image.mean()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        loss_per_image = loss_per_image.detach().cpu().numpy().tolist()

        return {"loss": loss_per_image}

    def validate_on_batch(self, data: dict) -> dict:
        images = data["image"].to(self.device)
        masks = data["mask"].to(self.device)
        image_spacings = data["image_spacing"]
        image_ids = data["id"]

        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = self.model(images)
            losses = self.loss_function(outputs, masks)
            loss_per_image = losses.mean(dim=(1, 2, 3, 4))
            outputs = torch.sigmoid(outputs)

        loss_per_image = loss_per_image.detach().cpu().numpy().tolist()
        outputs = outputs.detach().cpu().numpy().squeeze(axis=1)
        images = images.detach().cpu().numpy().squeeze(axis=1)
        masks = masks.detach().cpu().numpy().squeeze(axis=1)

        # image-wise evaluation and plots
        dices = []
        plots = []
        for output, image, mask, image_spacing, image_id in zip(
            outputs, images, masks, image_spacings, image_ids
        ):

            dice = dice_coefficient(
                prediction=output > 0.5,
                ground_truth=mask,
                # image_spacing=image_spacing,
            )
            dices.append(dice)

            # mid z slice
            mid_z_slice = image.shape[-1] // 2
            # Turn interactive plotting off
            plt.ioff()
            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
            ax[0].imshow(image[..., mid_z_slice], clim=(0, 0.5))
            ax[1].imshow(mask[..., mid_z_slice], clim=(0, 1))
            ax[2].imshow(output[..., mid_z_slice], clim=(0, 1))

            ax[0].set_title("image")
            ax[1].set_title("ground truth")
            ax[2].set_title("prediction")
            fig.suptitle(image_id)

            plots.append(Image(fig))
            # fig = px.imshow(image[..., mid_z_slice], zmin=0.0, zmax=0.5)
            #
            # self.aim_run.track(Figure(fig), name='FIG')

            # plots.append(Figure(fig))
            plt.close(fig)
            # Turn interactive plotting on again
            plt.ion()

        return {"loss": loss_per_image, "dice": dices, "prediction": plots}


import logging

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(LogFormatter())
logging.basicConfig(handlers=[handler])

logging.getLogger(__name__).setLevel(logging.DEBUG)
logging.getLogger("vroc").setLevel(logging.DEBUG)


(
    images_train,
    images_test,
    masks_train,
    masks_test,
    labels_train,
    labels_test,
) = train_test_split(
    all_images, all_masks, all_labels, test_size=0.10, random_state=1337
)

train_dataset = LungCTSegmentationDataset(
    image_filepaths=images_train,
    mask_filepaths=masks_train,
    mask_labels=labels_train,
    patch_shape=(128, 128, 128),
    image_spacing_range=((0.5, 4.0), (0.5, 4.0), (0.5, 4.0)),
    patches_per_image=1.0,
)
val_dataset = LungCTSegmentationDataset(
    image_filepaths=images_test,
    mask_filepaths=masks_test,
    mask_labels=labels_test,
    patch_shape=(320, 320, 160),
    image_spacing_range=None,
    patches_per_image=1,
    random_rotation=False,
    center_crop=True,
)


model = LungCTSegmentationUnet3d(n_levels=4)
optimizer = optim.Adam(model.parameters())

COLLATE_NOOP_KEYS = (
    "image_spacing",
    "full_image_shape",
    "i_patch",
    "n_patches",
    "patch_slicing",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    num_workers=4,
    collate_fn=partial(dict_collate, noop_keys=COLLATE_NOOP_KEYS),
)


val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    collate_fn=partial(dict_collate, noop_keys=COLLATE_NOOP_KEYS),
)
trainer = LungCTSegmentationTrainer(
    model=model,
    loss_function=nn.BCEWithLogitsLoss(reduction="none"),
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    run_folder="/datalake/learn2reg/runs",
    experiment_name="lung_segmentation",
)
trainer.run(steps=100_000, validation_interval=1000)
