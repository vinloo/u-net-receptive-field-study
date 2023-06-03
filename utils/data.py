import glob
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class SegmentationDataset(Dataset):
    def __init__(self, name: str, subset: str):
        """
        Initializes a SegmentationDataset object.

        Args:
            name (str): The name of the dataset.
            subset (str): The subset of the dataset to use (train, val, or test).

        Raises:
            ValueError: If an invalid dataset name is provided.

        """
        self.name = name
        if name not in ALL_DATASETS:
            raise ValueError(f"Dataset {name} not found")

        self.n_labels = ALL_DATASETS[name]["n_labels"]
        self.labels = ALL_DATASETS[name]["labels"]

        if self.n_labels == 1:
            self.inputs = glob.glob(f"data/preprocessed/{name}/{subset}/*.png")
            self.masks = glob.glob(
                f"data/preprocessed/{name}/{subset}/masks/*.png")
        else:
            self.inputs = glob.glob(f"data/preprocessed/{name}/{subset}/*.png")
            self.inputs.sort()
            masks = glob.glob(
                f"data/preprocessed/{name}/{subset}/masks/**/*.png")
            masks.sort()
            self.masks = []
            for i in range(len(masks) // self.n_labels):
                self.masks.append(
                    tuple(masks[i * self.n_labels: (i + 1) * self.n_labels]))
            assert len(self.inputs) == len(self.masks)

        self.inputs_dtype = torch.float32
        self.masks_dtype = torch.float32

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.inputs)

    def __getitem__(self, index: int):
        """
        Returns the input and mask tensors for a given index.

        Args:
            index (int): The index of the input and mask tensors to return.

        Returns:
            tuple: A tuple containing the input and mask tensors.

        """
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
        y = torch.where(y > 0.5, torch.tensor(
            255, dtype=self.masks_dtype), torch.tensor(0, dtype=self.masks_dtype))
        return x, y


ALL_DATASETS = {
    # dummy dataset for architecture testing
    "dummy": {
        "labels": ["dummy"],
        "n_labels": 1,
    },
    "fetal_head": {
        "labels": ["head"],
        "n_labels": 1,
    },
    "fetal_head_2": {
        "labels": ["head"],
        "n_labels": 1,
    },
    "breast_cancer": {
        "labels": ["cancer"],
        "n_labels": 1,
    },
    "mouse_embryo": {
        "labels": ["body", "bv"],
        "n_labels": 2,
    },
    "covid": {
        "labels": ["infection"],
        "n_labels": 1,
    },
    "covid_2": {
        "labels": ["infection"],
        "n_labels": 1,
    },
    "pancreas": {
        "labels": ["pancreas"],
        "n_labels": 1,
    },
    "brain_tumor": {
        "labels": ["tumor"],
        "n_labels": 1,
    },
    "prostate": {
        "labels": ["prostate"],
        "n_labels": 1,
    },
    "kidney": {
        "labels": ["kidney"],
        "n_labels": 1,
    },
    "lungs": {
        "labels": ["lungs"],
        "n_labels": 1,
    },
    "thyroid": {
        "labels": ["thyroid"],
        "n_labels": 1,
    },
    "nerve": {
        "labels": ["nerve"],
        "n_labels": 1,
    },
    "shapes_a": {
        "labels": ["shapes"],
        "n_labels": 1,
    },
    "shapes_b": {
        "labels": ["shapes"],
        "n_labels": 1,
    },
}
