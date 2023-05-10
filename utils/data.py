from torch.utils.data import DataLoader, Dataset
import torch
from torchvision.io import read_image
from enum import Enum


class SegmentationDataset(Dataset):
    def __init__(self,
                 inputs: list,
                 masks: list,
                 n_labels: int = 1,
                 ):
        self.inputs = inputs
        self.masks = masks
        self.inputs_dtype = torch.float32
        self.masks_dtype = torch.float32
        self.n_labels = n_labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.n_labels == 1:
            input_id = self.inputs[index]
            target_id = self.masks[index]
            x = read_image(input_id).type(self.inputs_dtype)
            y = read_image(target_id).type(self.masks_dtype)
            return x, y
        else:
            assert self.labels > 1
            # TODO: implement multi-label segmentation
    

class Modality(Enum):
    ULTRASOUND = "us"
    CT = "ct"
    MRI = "mri"
    XRAY = "xray"


ALL_DATASETS = {
    "fetal_head": {
        "labels": ["head"],
        "n_labels": 1,
        "modality": Modality.ULTRASOUND,
    },
    "breast_cancer": {
        "labels": ["cancer"],
        "n_labels": 1,
        "modality": Modality.ULTRASOUND,
    },
    "mouse_embryo": {
        "labels": ["body", "bv"],
        "n_labels": 2,
        "modality": Modality.ULTRASOUND,
    },
    "covid_19": {
        "labels": ["ground-glass", "consolidation", "pleural-effusion"],
        "n_labels": 3,
        "modality": Modality.CT,
    },
    "pancreas": {
        "labels": ["pancreas"],
        "n_labels": 1,
        "modality": Modality.CT,
    },
    "brain_tumor": {
        "labels": ["tumor"],
        "n_labels": 1,
        "modality": Modality.MRI,
    },
    "prostate": {
        "labels": ["prostate"],
        "n_labels": 1,
        "modality": Modality.MRI,
    },
}
