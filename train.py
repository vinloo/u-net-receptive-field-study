import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import shutil
import json
from utils.data import SegmentationDataset
from utils.config import load_config, ALL_CONFIGS
from utils.data import ALL_DATASETS
from utils.files import get_output_path
from torchvision.io import read_image
from dotmap import DotMap
from unet import UNet
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from tqdm import tqdm, trange


class EarlyStopper:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        training_dataloader: Dataset,
        validation_dataloader: Optional[Dataset] = None,
        epochs: int = 100,
        epoch: int = 0,
        scheduler = None,
        no_progressbar: bool = False,
        attention: bool = False
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.center_trf = self.model.center_trf()

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.scheduler = scheduler
        self.no_progressbar = no_progressbar
        self.attention = attention

    def run_trainer(self, dataset_name, config_name, out_dir):
        out_path = get_output_path(dataset_name, config_name, out_dir, self.attention)
        early_stopper = EarlyStopper(patience=25)
        progressbar = trange(self.epochs, desc="Progress")
        for i in progressbar:
            self.epoch += 1
            self._train()
            self._validate()

            # only save if it performs better
            if self.validation_loss[-1] <= min(self.validation_loss):
                torch.save({
                    'epoch': i,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.validation_loss,
                }, f"{out_path}/best_model.pt")

            if early_stopper.early_stop(self.validation_loss[-1]):
                print(f"Stopping early because validation loss has not improved for {early_stopper.patience} epochs")
                break

        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):
        self.model.train()
        train_losses = []
        batch_iter = tqdm(
            enumerate(self.training_dataloader),
            "Training",
            total=len(self.training_dataloader),
            leave=False,
            disable=self.no_progressbar,
        )

        for _, (x, y) in batch_iter:
            input_x = x.to(self.device)
            input_x.requires_grad = True
            target_y =  y.to(self.device) / 255
            self.optimizer.zero_grad()
            out = self.model(input_x)
            loss = self.criterion(out, target_y)
            loss_value = loss.item()
            train_losses.append(abs(loss_value))
            loss.backward()
            self.optimizer.step()

            batch_iter.set_description(
                f"Training: (loss {loss_value:.4f})"
            )

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]["lr"])
        self.scheduler.step(np.mean(train_losses))

        batch_iter.close()

    def _validate(self):
        self.model.eval()
        valid_losses = []
        batch_iter = tqdm(
            enumerate(self.validation_dataloader),
            "Validation",
            total=len(self.validation_dataloader),
            leave=False,
            disable=self.no_progressbar,
        )

        for _, (x, y) in batch_iter:
            input = x.to(self.device)
            target = y.to(self.device) / 255

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(abs(loss_value))

                batch_iter.set_description(
                    f"Validation: (loss {loss_value:.4f})")

        self.validation_loss.append(np.mean(valid_losses))
        batch_iter.close()


def train(config: DotMap, dataset_name: str, config_name: str, n_epochs=10, batch_size=1, lr=0.01, out_dir="out", no_progress=False, attention=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    dataset_train = SegmentationDataset(dataset_name, "train")
    dataset_val = SegmentationDataset(dataset_name, "val")
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    model = UNet(config, attention, dataset_train.n_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-9)

    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        training_dataloader=dataloader_train,
        validation_dataloader=dataloader_val,
        epochs=n_epochs,
        scheduler=scheduler,
        no_progressbar=no_progress,
        attention=attention
    )

    get_output_path(dataset_name, config_name, out_dir, attention, clear_existing=True)
    trainer.run_trainer(dataset_name, config_name, out_dir)

    print("Training finished")
    print("Lowest validation loss at epoch", trainer.validation_loss.index(min(trainer.validation_loss)))

    out_path = get_output_path(dataset_name, config_name, out_dir, attention)

    plt.plot(trainer.training_loss, label="Training loss")
    plt.plot(trainer.validation_loss, label="Validation loss")
    plt.title(f"Loss for {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{out_path}/result.png')

    results = {
        "dataset": dataset_name,
        "config": config_name,
        "training_loss": trainer.training_loss,
        "validation_loss": trainer.validation_loss,
        "learning_rate": trainer.learning_rate,
        "training_time": int(np.argmin(trainer.validation_loss) + 1),
    }

    # write results to json
    with open(f'{out_path}/result.json', 'w') as f:
        json.dump(results, f)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="dataset to preprocess", choices=ALL_DATASETS.keys(), required=True)
    parser.add_argument("-c", "--config", type=str, help="path to config file", choices=ALL_CONFIGS, required=True)
    parser.add_argument("-e", "--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("-o", "--output_dir", type=str, default="out", help="output folder")
    parser.add_argument("-n", "--no_progress", action="store_true", help="disable progress bar")
    parser.add_argument("-a", "--attention", action="store_true", help="attention u-net")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, args.dataset, args.config, n_epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate, out_dir=args.output_dir, no_progress=args.no_progress, attention=args.attention)
    

if __name__ == "__main__":
    main()
