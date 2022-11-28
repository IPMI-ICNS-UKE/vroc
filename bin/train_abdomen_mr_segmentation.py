from functools import partial, reduce
from operator import add
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from aim import Image
from monai import losses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from vroc.dataset import SegmentationDataset
from vroc.helper import dict_collate
from vroc.logger import init_fancy_logging
from vroc.models import Unet3d
from vroc.trainer import BaseTrainer, MetricType


def combine_datasets(datasets: dict) -> dict:
    all_images = reduce(add, [d["images"] for d in datasets.values()])
    all_masks = reduce(add, [d["masks"] for d in datasets.values()])
    all_labels = reduce(
        add, [[d.get("labels")] * len(d["masks"]) for d in datasets.values()]
    )

    return {"images": all_images, "masks": all_masks, "labels": all_labels}


def train_test_split_dataset(dataset: dict, **kwargs) -> Tuple[dict, dict]:
    n_samples = len(dataset["images"])
    sample_idx = np.arange(n_samples)
    train_sample_idx, test_sample_idx = train_test_split(sample_idx, **kwargs)

    train_sample_idx = set(train_sample_idx)
    test_sample_idx = set(test_sample_idx)

    train_dataset = {
        key: [v for i, v in enumerate(val) if i in train_sample_idx]
        for (key, val) in dataset.items()
    }
    test_dataset = {
        key: [v for i, v in enumerate(val) if i in test_sample_idx]
        for (key, val) in dataset.items()
    }
    return train_dataset, test_dataset


datasets = {
    "oasis": {
        "images": sorted(
            Path("/datalake/learn2reg/AbdomenMRMR/imagesTr").glob("*.nii.gz")
        ),
        "masks": sorted(
            Path("/datalake/learn2reg/AbdomenMRMR/labelsTr").glob("*.nii.gz")
        ),
        "labels": None,  # do not merge any labels, handle as distinct labels
    },
}
dataset = combine_datasets(datasets)

train_dataset, test_dataset = train_test_split_dataset(
    dataset, test_size=0.25, random_state=1337
)


class AbdomenMRSegmentationTrainer(BaseTrainer):
    METRICS = {
        "loss": MetricType.SMALLER_IS_BETTER,
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
        image_filenames = data["image_filename"]

        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = self.model(images)
            losses = self.loss_function(outputs, masks)
            loss_per_image = losses.mean(dim=(1, 2, 3, 4))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1, keepdim=True)

        loss_per_image = loss_per_image.detach().cpu().numpy().tolist()
        outputs = outputs.detach().cpu().numpy().squeeze(axis=1)
        images = images.detach().cpu().numpy().squeeze(axis=1)
        masks = masks.detach().cpu().numpy().squeeze(axis=1)

        # image-wise evaluation and plots
        plots = []
        for output, image, mask, image_spacing, image_id, image_filename in zip(
            outputs, images, masks, image_spacings, image_ids, image_filenames
        ):
            # mid z slice
            mid_z_slice = image.shape[-1] // 2
            # Turn interactive plotting off
            with plt.ioff():
                fig, ax = plt.subplots(
                    1, 3, sharex=True, sharey=True, figsize=(12, 9), dpi=300
                )
                ax[0].imshow(image[..., mid_z_slice], clim=(0, 0.5))
                ax[1].imshow(mask[..., mid_z_slice], cmap="nipy_spectral")
                ax[2].imshow(output[..., mid_z_slice], cmap="nipy_spectral")

                ax[0].set_title("image")
                ax[1].set_title("ground truth")
                ax[2].set_title("prediction")
                fig.suptitle(image_filename)

                plots.append(Image(fig))
                plt.close(fig)

        return {"loss": loss_per_image, "prediction": plots}


import logging

init_fancy_logging()

logging.getLogger(__name__).setLevel(logging.INFO)
logging.getLogger("vroc").setLevel(logging.INFO)


train_dataset = SegmentationDataset(
    image_filepaths=train_dataset["images"],
    segmentation_filepaths=train_dataset["masks"],
    segmentation_labels=train_dataset["labels"],
    patch_shape=(128, 128, 128),
    image_spacing_range=((0.5, 4.0), (0.5, 4.0), (0.5, 4.0)),
    patches_per_image=1.0,
    input_value_range=(0, 1),
    output_value_range=(0, 1),
)
val_dataset = SegmentationDataset(
    image_filepaths=test_dataset["images"],
    segmentation_filepaths=test_dataset["masks"],
    segmentation_labels=test_dataset["labels"],
    patch_shape=(192, 160, 192),
    image_spacing_range=None,
    patches_per_image=1,
    random_rotation=False,
    center_crop=True,
    input_value_range=(0, 1),
    output_value_range=(0, 1),
)


model = Unet3d(n_classes=5, n_levels=4, filter_base=16)
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

loss_function = losses.DiceLoss(
    include_background=True, to_onehot_y=True, reduction="none", softmax=True
)

trainer = AbdomenMRSegmentationTrainer(
    model=model,
    loss_function=loss_function,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    run_folder="/datalake/learn2reg/runs",
    experiment_name="abdomen_mr_segmentation",
    device="cuda:0",
)
trainer.run(steps=100_000, validation_interval=1000)
