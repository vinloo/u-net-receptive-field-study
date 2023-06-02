import glob
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from enum import Enum


class SegmentationDataset(Dataset):
    def __init__(self, name: str, subset: str):
        self.name = name
        if name not in ALL_DATASETS:
            raise ValueError(f"Dataset {name} not found")
        
        self.n_labels = ALL_DATASETS[name]["n_labels"]
        self.labels = ALL_DATASETS[name]["labels"]
        self.modality = ALL_DATASETS[name]["modality"]
    
        if self.n_labels == 1:
            self.inputs = glob.glob(f"data/preprocessed/{name}/{subset}/*.png")
            self.masks = glob.glob(f"data/preprocessed/{name}/{subset}/masks/*.png")
        else:
            self.inputs = glob.glob(f"data/preprocessed/{name}/{subset}/*.png")
            self.inputs.sort()
            masks = glob.glob(f"data/preprocessed/{name}/{subset}/masks/**/*.png")
            masks.sort()
            self.masks = []
            for i in range(len(masks) // self.n_labels):
                self.masks.append(tuple(masks[i * self.n_labels: (i + 1) * self.n_labels]))
            assert len(self.inputs) == len(self.masks)

        self.inputs_dtype = torch.float32
        self.masks_dtype = torch.float32

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.n_labels == 1:
            input_id = self.inputs[index]
            target_id = self.masks[index]
            x = read_image(input_id).type(self.inputs_dtype)
            y = read_image(target_id).type(self.masks_dtype)
        else:
            assert self.n_labels > 1
            input_id = self.inputs[index]
            target_ids = self.masks[index]
            x = read_image(input_id).type(self.inputs_dtype)
            y = []
            for i in range(self.n_labels):
                y.append(read_image(target_ids[i]).type(self.masks_dtype))
            y = torch.cat(y, dim=0)
            
        # treshold y to only 0 and 255
        y = torch.where(y > 0.5, torch.tensor(255, dtype=self.masks_dtype), torch.tensor(0, dtype=self.masks_dtype))
        return x, y
    

class Modality(Enum):
    ULTRASOUND = "us"
    CT = "ct"
    MRI = "mri"
    XRAY = "xray"


ALL_DATASETS = {
    # dummy dataset for architecture testing
    "dummy": {
        "labels": ["dummy"],
        "n_labels": 1,
        "modality": None,
    },
    "fetal_head": {
        "labels": ["head"],
        "n_labels": 1,
        "modality": Modality.ULTRASOUND,
    },
    "fetal_head_2": {
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
    "covid": {
        "labels": ["infection"],
        "n_labels": 1,
        "modality": Modality.CT,
    },
    "covid_2": {
        "labels": ["infection"],
        "n_labels": 1,
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
    "kidney": {
        "labels": ["kidney"],
        "n_labels": 1,
        "modality": Modality.MRI,
    },
    "lungs": {
        "labels": ["lungs"],
        "n_labels": 1,
        "modality": Modality.XRAY,
    },
    "thyroid": {
        "labels": ["thyroid"],
        "n_labels": 1,
        "modality": Modality.ULTRASOUND,
    },
    "nerve": {
        "labels": ["nerve"],
        "n_labels": 1,
        "modality": Modality.ULTRASOUND,
    },
    "shapes_a": { 
        "labels": ["shapes"],
        "n_labels": 1,
        "modality": None,
    },
    "shapes_b": {
        "labels": ["shapes"],
        "n_labels": 1,
        "modality": None,
    },
}
