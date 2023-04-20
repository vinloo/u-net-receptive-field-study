from torch.utils.data import DataLoader, Dataset
import torch
from torchvision.io import read_image


class SegmentationDataset(Dataset):
    def __init__(self,
                 inputs: list,
                 masks: list,
                 ):
        self.inputs = inputs
        self.masks = masks
        self.inputs_dtype = torch.float32
        self.masks_dtype = torch.float32

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        input_id = self.inputs[index]
        target_id = self.masks[index]
        x = read_image(input_id).type(self.inputs_dtype)
        y = read_image(target_id).type(self.masks_dtype)

        return x, y