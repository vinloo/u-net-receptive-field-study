import glob
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import shutil
from preprocess_data import ALL_DATASETS
from torchvision.io import read_image
from dotmap import DotMap
from PIL import Image
from unet import UNet
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from tqdm import tqdm, trange


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
        scheduler = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.scheduler = scheduler

    def run_trainer(self, dataset_name, config_name):
        chkp_dir = f"checkpoints/{dataset_name}/{config_name}"
        if not os.path.exists(chkp_dir):
            os.makedirs(chkp_dir)
        else:
            shutil.rmtree(chkp_dir)
            os.makedirs(chkp_dir)

        progressbar = trange(self.epochs, desc="Progress")
        for i in progressbar:
            self.epoch += 1
            self._train()
            self._validate()
            torch.save({
                'epoch': i,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.validation_loss,
            }, f"{chkp_dir}/checkpoint_{i}_{self.validation_loss[-1]:.2f}.pt")

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

        # https://lars76.github.io/2021/09/05/activations-segmentation.html
        def _activation(x):
            return (0.5 - 1e-7) * torch.erf(x/torch.sqrt(torch.tensor(2))) + 0.5

        for _, (x, y) in batch_iter:
            input_x = x.to(self.device)
            target_y =  y.to(self.device) / 255
            self.optimizer.zero_grad()
            out = self.model(input_x)
            # print(out.min(), out.max(), target_y.min(), target_y.max())
            # out = _activation(out)
            # print(str(out[0,0,:,:]))
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


def train(config: DotMap, dataset_name: str, config_name: str, n_epochs=10, batch_size=1, lr=0.001, seed=42):
    torch.manual_seed(seed)

    # TODO: Make code work for batch size >1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    inputs_train = glob.glob(os.path.join(f"data/preprocessed/{dataset_name}/train", "*.png"))
    masks_train = glob.glob(os.path.join(
        f"data/preprocessed/{dataset_name}/train/masks", "*.png"))
    inputs_val = glob.glob(os.path.join(f"data/preprocessed/{dataset_name}/val", "*.png"))
    masks_val = glob.glob(os.path.join(
        f"data/preprocessed/{dataset_name}/val/masks", "*.png"))

    dataset_train = SegmentationDataset(inputs_train, masks_train)
    dataset_val = SegmentationDataset(inputs_val, masks_val)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    model = UNet(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-9)
    # criterion = torch.nn.BCELoss(reduction='mean')

    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        training_dataloader=dataloader_train,
        validation_dataloader=dataloader_val,
        epochs=n_epochs,
        scheduler=scheduler
    )

    trainer.run_trainer(dataset_name, config_name)

    print("Training finished")
    print("Lowest validation loss at epoch", trainer.validation_loss.index(min(trainer.validation_loss)))

    plt.plot(trainer.training_loss, label="Training loss")
    plt.plot(trainer.validation_loss, label="Validation loss")
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="dataset to preprocess", choices=ALL_DATASETS, required=True)
    parser.add_argument("-s", "--seed", type=int, default=42, help="random seed")
    parser.add_argument("-c", "--config", type=str, default="configurations/default.json", help="path to config file")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate")
    args = parser.parse_args()

    # validate config
    try:
        config = json.load(open(args.config))
        config = DotMap(config, _dynamic=False)
    except json.decoder.JSONDecodeError:
        print("Invalid config file")
        exit(1)

    try:
        assert type(config.grayscale) is bool, "config.grayscale is of the wrong type, expected bool"

        assert type(config.enc1.conv1.k) is int, "config.enc1.conv1.k is of the wrong type, expected int"
        assert type(config.enc1.conv1.s) is int, "config.enc1.conv1.s is of the wrong type, expected int"
        assert type(config.enc1.conv1.p) is int, "config.enc1.conv1.p is of the wrong type, expected int"
        assert type(config.enc1.conv2.k) is int, "config.enc1.conv2.k is of the wrong type, expected int"
        assert type(config.enc1.conv2.s) is int, "config.enc1.conv2.s is of the wrong type, expected int"
        assert type(config.enc1.conv2.p) is int, "config.enc1.conv2.p is of the wrong type, expected int"
        assert type(config.enc1.pool_k) is int, "config.enc1.pool_k is of the wrong type, expected int"

        assert type(config.enc2.conv1.k) is int, "config.enc2.conv1.k is of the wrong type, expected int"
        assert type(config.enc2.conv1.s) is int, "config.enc2.conv1.s is of the wrong type, expected int"
        assert type(config.enc2.conv1.p) is int, "config.enc2.conv1.p is of the wrong type, expected int"
        assert type(config.enc2.conv2.k) is int, "config.enc2.conv2.k is of the wrong type, expected int"
        assert type(config.enc2.conv2.s) is int, "config.enc2.conv2.s is of the wrong type, expected int"
        assert type(config.enc2.conv2.p) is int, "config.enc2.conv2.p is of the wrong type, expected int"
        assert type(config.enc2.pool_k) is int, "config.enc2.pool_k is of the wrong type, expected int"

        assert type(config.enc3.conv1.k) is int, "config.enc3.conv1.k is of the wrong type, expected int"
        assert type(config.enc3.conv1.s) is int, "config.enc3.conv1.s is of the wrong type, expected int"
        assert type(config.enc3.conv1.p) is int, "config.enc3.conv1.p is of the wrong type, expected int"
        assert type(config.enc3.conv2.k) is int, "config.enc3.conv2.k is of the wrong type, expected int"
        assert type(config.enc3.conv2.s) is int, "config.enc3.conv2.s is of the wrong type, expected int"
        assert type(config.enc3.conv2.p) is int, "config.enc3.conv2.p is of the wrong type, expected int"
        assert type(config.enc3.pool_k) is int, "config.enc3.pool_k is of the wrong type, expected int"

        assert type(config.enc4.conv1.k) is int, "config.enc4.conv1.k is of the wrong type, expected int"
        assert type(config.enc4.conv1.s) is int, "config.enc4.conv1.s is of the wrong type, expected int"
        assert type(config.enc4.conv1.p) is int, "config.enc4.conv1.p is of the wrong type, expected int"
        assert type(config.enc4.conv2.k) is int, "config.enc4.conv2.k is of the wrong type, expected int"
        assert type(config.enc4.conv2.s) is int, "config.enc4.conv2.s is of the wrong type, expected int"
        assert type(config.enc4.conv2.p) is int, "config.enc4.conv2.p is of the wrong type, expected int"
        assert type(config.enc4.pool_k) is int, "config.enc4.pool_k is of the wrong type, expected int"

        assert type(config.b.conv1.k) is int, "config.b.conv1.k is of the wrong type, expected int"
        assert type(config.b.conv1.s) is int, "config.b.conv1.s is of the wrong type, expected int"
        assert type(config.b.conv1.p) is int, "config.b.conv1.p is of the wrong type, expected int"
        assert type(config.b.conv2.k) is int, "config.b.conv2.k is of the wrong type, expected int"
        assert type(config.b.conv2.s) is int, "config.b.conv2.s is of the wrong type, expected int"
        assert type(config.b.conv2.p) is int, "config.b.conv2.p is of the wrong type, expected int"

        assert type(config.dec1.up.k) is int, "config.dec1.up.k is of the wrong type, expected int"
        assert type(config.dec1.up.s) is int, "config.dec1.up.s is of the wrong type, expected int"
        assert type(config.dec1.up.p) is int, "config.dec1.up.p is of the wrong type, expected int"
        assert type(config.dec1.conv1.k) is int, "config.dec1.conv1.k is of the wrong type, expected int"
        assert type(config.dec1.conv1.s) is int, "config.dec1.conv1.s is of the wrong type, expected int"
        assert type(config.dec1.conv1.p) is int, "config.dec1.conv1.p is of the wrong type, expected int"
        assert type(config.dec1.conv2.k) is int, "config.dec1.conv2.k is of the wrong type, expected int"
        assert type(config.dec1.conv2.s) is int, "config.dec1.conv2.s is of the wrong type, expected int"
        assert type(config.dec1.conv2.p) is int, "config.dec1.conv2.p is of the wrong type, expected int"

        assert type(config.dec2.up.k) is int, "config.dec2.up.k is of the wrong type, expected int"
        assert type(config.dec2.up.s) is int, "config.dec2.up.s is of the wrong type, expected int"
        assert type(config.dec2.up.p) is int, "config.dec2.up.p is of the wrong type, expected int"
        assert type(config.dec2.conv1.k) is int, "config.dec2.conv1.k is of the wrong type, expected int"
        assert type(config.dec2.conv1.s) is int, "config.dec2.conv1.s is of the wrong type, expected int"
        assert type(config.dec2.conv1.p) is int, "config.dec2.conv1.p is of the wrong type, expected int"
        assert type(config.dec2.conv2.k) is int, "config.dec2.conv2.k is of the wrong type, expected int"
        assert type(config.dec2.conv2.s) is int, "config.dec2.conv2.s is of the wrong type, expected int"
        assert type(config.dec2.conv2.p) is int, "config.dec2.conv2.p is of the wrong type, expected int"

        assert type(config.dec3.up.k) is int, "config.dec3.up.k is of the wrong type, expected int"
        assert type(config.dec3.up.s) is int, "config.dec3.up.s is of the wrong type, expected int"
        assert type(config.dec3.up.p) is int, "config.dec3.up.p is of the wrong type, expected int"
        assert type(config.dec3.conv1.k) is int, "config.dec3.conv1.k is of the wrong type, expected int"
        assert type(config.dec3.conv1.s) is int, "config.dec3.conv1.s is of the wrong type, expected int"
        assert type(config.dec3.conv1.p) is int, "config.dec3.conv1.p is of the wrong type, expected int"
        assert type(config.dec3.conv2.k) is int, "config.dec3.conv2.k is of the wrong type, expected int"
        assert type(config.dec3.conv2.s) is int, "config.dec3.conv2.s is of the wrong type, expected int"
        assert type(config.dec3.conv2.p) is int, "config.dec3.conv2.p is of the wrong type, expected int"

        assert type(config.dec4.up.k) is int, "config.dec4.up.k is of the wrong type, expected int"
        assert type(config.dec4.up.s) is int, "config.dec4.up.s is of the wrong type, expected int"
        assert type(config.dec4.up.p) is int, "config.dec4.up.p is of the wrong type, expected int"
        assert type(config.dec4.conv1.k) is int, "config.dec4.conv1.k is of the wrong type, expected int"
        assert type(config.dec4.conv1.s) is int, "config.dec4.conv1.s is of the wrong type, expected int"
        assert type(config.dec4.conv1.p) is int, "config.dec4.conv1.p is of the wrong type, expected int"
        assert type(config.dec4.conv2.k) is int, "config.dec4.conv2.k is of the wrong type, expected int"
        assert type(config.dec4.conv2.s) is int, "config.dec4.conv2.s is of the wrong type, expected int"
        assert type(config.dec4.conv2.p) is int, "config.dec4.conv2.p is of the wrong type, expected int"

        assert type(config.out.k) is int, "config.out.k is of the wrong type, expected int"
        assert type(config.out.s) is int, "config.out.s is of the wrong type, expected int"
        assert type(config.out.p) is int, "config.out.p is of the wrong type, expected int"

    except AttributeError as e:
        print(f"'{args.config}' is missing attribute '{e.name}'")
        exit(1)
    except AssertionError as e:
        print(e)
        exit(1)

    config_name = args.config.split("/")[-1].split(".")[0]

    train(config, args.dataset, config_name, n_epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate, seed=args.seed)
    

if __name__ == "__main__":
    main()
