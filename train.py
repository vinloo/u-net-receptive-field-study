import glob
import os
import torch
from PIL import Image
from unet import UNet
from torch.utils.data import DataLoader, Dataset
from configurations.default import config as default_config
from typing import Optional
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import numpy as np

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

    def __getitem__(self,
                    index: int):
        input_id = self.inputs[index]
        target_id = self.masks[index]

        x, y = Image.open(input_id), Image.open(target_id)

        # convert to grayscale and scale to fixed size
        x = x.convert('L')
        y = y.convert('L')
        x = x.resize((576, 576))
        y = y.resize((576, 576))

        x = np.array(x)
        y = np.array(y)

        x, y = torch.from_numpy(x).type(self.inputs_dtype).unsqueeze(0), torch.from_numpy(y).type(self.masks_dtype).unsqueeze(0)

        return x, y


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        training_dataloader: Dataset,
        validation_dataloader: Optional[Dataset] = None,
        lr_scheduler = None,
        epochs: int = 100,
        epoch: int = 0,
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):          

        progressbar = trange(self.epochs, desc="Progress")
        for i in progressbar:
            self.epoch += 1
            self._train()
            self._validate()

            if self.lr_scheduler is not None:
                if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.lr_scheduler.batch(
                        self.validation_loss[i]
                    )  
                else:
                    self.lr_scheduler.batch()
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        self.model.train()
        train_losses = []
        batch_iter = tqdm(
            enumerate(self.training_dataloader),
            "Training",
            total=len(self.training_dataloader),
            leave=False,
        )

        for _, (x, y) in batch_iter:
            input_x, target_y = x.to(self.device), y.to(
                self.device
            ) 
            self.optimizer.zero_grad()
            out = self.model(input_x)
            # print(str(out[0,0,:,:]))
            loss = self.criterion(out[0,0,:,:], target_y[0,0,:,:])
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()
            self.optimizer.step() 

            batch_iter.set_description(
                f"Training: (loss {loss_value:.4f})"
            )

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]["lr"])

        batch_iter.close()

    def _validate(self):

        self.model.eval()
        valid_losses = []
        batch_iter = tqdm(
            enumerate(self.validation_dataloader),
            "Validation",
            total=len(self.validation_dataloader),
            leave=False,
        )

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(
                self.device
            ) 

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out[0,0,:,:], target[0,0,:,:])
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f"Validation: (loss {loss_value:.4f})")

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()


class BinaryDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(2) + target.pow(2), dim=1) + self.smooth
        loss = 1 - num / den
        return loss.mean()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    inputs_train = glob.glob(os.path.join("data/preprocessed/train", "*.png"))
    masks_train = glob.glob(os.path.join("data/preprocessed/train/masks", "*.png"))
    inputs_val = glob.glob(os.path.join("data/preprocessed/train", "*.png"))
    masks_val = glob.glob(os.path.join("data/preprocessed/train/masks", "*.png"))


    dataset_train = SegmentationDataset(inputs_train, masks_val)
    dataset_val = SegmentationDataset(inputs_val, masks_val)
    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True)


    model = UNet(default_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = BinaryDiceLoss()

    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        training_dataloader=dataloader_val,
        validation_dataloader=dataloader_val,
        epochs=10
    )

    trainer.run_trainer()

    plt.plot(trainer.training_loss, label="Training loss")
    plt.plot(trainer.validation_loss, label="Validation loss")
    plt.legend()
    plt.show()





