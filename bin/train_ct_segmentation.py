import json
import re
from functools import partial, reduce
from operator import add
from pathlib import Path
from typing import List

import torch
import torch.optim as optim
from monai import losses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from vroc.dataset import SegmentationDataset
from vroc.helper import dict_collate
from vroc.logger import init_fancy_logging
from vroc.models import Unet3d
from vroc.trainer import BaseTrainer, MetricType


def match_labels(labels: dict[int, str], pattern: str) -> List[int]:
    members = []
    for label_id, label_name in labels.items():
        if re.match(pattern, label_name):
            members.append(label_id)

    return members


datasets = {
    "totalsegmentator": {
        "images": sorted(Path("/datalake/totalsegmentator").glob("s*/ct.nii.gz")),
        "masks": sorted(
            Path("/datalake/totalsegmentator").glob("s*/merged_segmentations.nii.gz")
        ),
    },
}


all_images = reduce(add, [d["images"] for d in datasets.values()])
all_masks = reduce(add, [d["masks"] for d in datasets.values()])
all_labels = reduce(
    add, [[d.get("labels")] * len(d["masks"]) for d in datasets.values()]
)

label_names = {0: "background"}
with open("/datalake/totalsegmentator/classes.json", "rt") as f:
    label_names.update(json.load(f))
label_names = {int(k): v.split(".")[0] for (k, v) in label_names.items()}


# merge labels
skeleton = [
    "clavicula_.*",
    "humerus_.*",
    "scapula_.*",
    "rib_.*",
    "vertebrae_.*",
    "hip_.*",
    "sacrum",
    "femur_.*",
]
skeleton = {
    pattern.replace("_.*", ""): match_labels(label_names, pattern=pattern)
    for pattern in skeleton
}

# overview of all labels in publication: https://arxiv.org/pdf/2208.05868.pdf
names = list(label_names.values())
merging = {
    "ribs": tuple(range(49, 73)),
    "vertebrae": tuple(range(81, 105)),
    # ignore
    "background": (
        names.index("background"),
        names.index("aorta"),
        names.index("autochthon_left"),
        names.index("autochthon_right"),
        names.index("face"),
        names.index("gluteus_maximus_left"),
        names.index("gluteus_maximus_right"),
        names.index("gluteus_medius_left"),
        names.index("gluteus_medius_right"),
        names.index("gluteus_minimus_left"),
        names.index("gluteus_minimus_right"),
        names.index("heart_atrium_left"),
        names.index("heart_atrium_right"),
        names.index("heart_myocardium"),
        names.index("heart_ventricle_left"),
        names.index("heart_ventricle_right"),
        names.index("iliac_artery_left"),
        names.index("iliac_artery_right"),
        names.index("iliac_vena_left"),
        names.index("iliac_vena_right"),
        names.index("iliopsoas_left"),
        names.index("iliopsoas_right"),
        names.index("inferior_vena_cava"),
        names.index("portal_vein_and_splenic_vein"),
        names.index("pulmonary_artery"),
    ),
}

merging = {_v: k for k, v in merging.items() for _v in v}

merged_label_names = {k: merging.get(k) or v for (k, v) in label_names.items()}


label_id_to_name = dict.fromkeys(merged_label_names.values())
label_id_to_name = {i: k for i, k in enumerate(label_id_to_name)}

label_name_to_id = {v: k for (k, v) in label_id_to_name.items()}


old_to_new_label_id = {}
for (old_id, old_name) in label_names.items():
    if old_name in label_name_to_id:
        old_to_new_label_id[old_id] = label_name_to_id[old_name]
    elif old_id in merging:
        new_name = merging[old_id]
        old_to_new_label_id[old_id] = label_name_to_id[new_name]
    else:
        old_to_new_label_id[old_id] = None


class CTSegmentationTrainer(BaseTrainer):
    METRICS = {"loss": MetricType.SMALLER_IS_BETTER}

    def __init__(self, *args, label_names: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = label_names or {}

    def train_on_batch(self, data: dict) -> dict:

        images = data["image"].to(self.device)
        segmentations = data["segmentation"].to(self.device)

        for old_label_id, new_label_id in old_to_new_label_id.items():
            segmentations[segmentations == old_label_id] = new_label_id

        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = self.model(images)
            losses = self.loss_function(outputs, segmentations)
            loss_per_image = losses.mean(dim=(1, 2, 3, 4))
            loss = loss_per_image.mean()
            loss_per_label = losses.mean(dim=(0, 2, 3, 4))

        loss_per_image = loss_per_image.detach().cpu().numpy().tolist()
        loss_per_label = loss_per_label.detach().cpu().numpy().tolist()

        label_losses = {
            f"loss_label_{self.label_names.get(i, i)}": loss_per_label[i]
            for i in range(len(loss_per_label))
        }

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {**label_losses, "loss": loss_per_image}

    def validate_on_batch(self, data: dict) -> dict:
        images = data["image"].to(self.device)
        segmentations = data["segmentation"].to(self.device)

        for old_label_id, new_label_id in old_to_new_label_id.items():
            segmentations[segmentations == old_label_id] = new_label_id

        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = self.model(images)
            losses = self.loss_function(outputs, segmentations)
            loss_per_image = losses.mean(dim=(1, 2, 3, 4))
            loss_per_label = losses.mean(dim=(0, 2, 3, 4))

        loss_per_image = loss_per_image.detach().cpu().numpy().tolist()
        loss_per_label = loss_per_label.detach().cpu().numpy().tolist()

        label_losses = {
            f"loss_label_{self.label_names.get(i, i)}": loss_per_label[i]
            for i in range(len(loss_per_label))
        }

        return {**label_losses, "loss": loss_per_image}


import logging

init_fancy_logging()

logging.getLogger(__name__).setLevel(logging.DEBUG)
logging.getLogger("vroc").setLevel(logging.DEBUG)

DEVICE = "cuda:0"

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

train_dataset = SegmentationDataset(
    image_filepaths=images_train,
    segmentation_filepaths=masks_train,
    segmentation_labels=labels_train,
    patch_shape=(128, 128, 128),
    image_spacing_range=None,  # ((1.5, 1.5), (1.5, 1.5), (1.5, 1.5)),
    patches_per_image=1.0,
    random_rotation=False,
    input_value_range=(-1024, 3071),
    output_value_range=(0, 1),
)

val_dataset = SegmentationDataset(
    image_filepaths=images_test,
    segmentation_filepaths=masks_test,
    segmentation_labels=labels_test,
    patch_shape=(256, 256, 192),
    image_spacing_range=None,
    patches_per_image=8,
    random_rotation=False,
    center_crop=False,
    input_value_range=(-1024, 3071),
    output_value_range=(0, 1),
)


n_classes = len(label_id_to_name)
encoder_n_filters = (n_classes, 2 * n_classes, 4 * n_classes, 4 * n_classes)
decoder_n_filters = (4 * n_classes, 4 * n_classes, 2 * n_classes, 2 * n_classes)

print(f"{encoder_n_filters=}")
print(f"{decoder_n_filters=}")

model = Unet3d(
    n_classes=n_classes,
    n_levels=4,
    n_filters=(n_classes, *encoder_n_filters, *decoder_n_filters, 2 * n_classes),
)
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
    batch_size=4,
    num_workers=8,
    prefetch_factor=1,
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

trainer = CTSegmentationTrainer(
    label_names=label_id_to_name,
    model=model,
    loss_function=loss_function,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    run_folder="/datalake/learn2reg/runs",
    experiment_name="ct_segmentation",
    device=DEVICE,
)
trainer.run(steps=1_000_000, validation_interval=1000)

# import SimpleITK as sitk
# from vroc.segmentation import MultiClassSegmenter3d
# image = sitk.ReadImage(str(images_test[0]))
# image_arr = sitk.GetArrayFromImage(image)
# image_arr = image_arr.swapaxes(0, 2)
#
# state = model.load_state_dict()
#
# segmenter = MultiClassSegmenter3d(
#     model=model,
#     input_value_range=(-1024, 3071),
#     output_value_range=(0, 1),
#     pad_to_pow_2=True,
#     device='cuda:0'
# )
#
# segmentation = segmenter.segment(image=image_arr)
