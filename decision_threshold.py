import torch
import numpy as np
import os
import pandas as pd
import json
import argparse
import datetime
from tqdm import tqdm
from unet import UNet
from torch.utils.data import DataLoader
from utils.metrics import specificity, sensitivity
from utils.config import load_config
from utils.data import SegmentationDataset
from utils.data import ALL_DATASETS
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def model_decision_threshold(config_name, dataset_name, device="cuda", N=100):
    config = load_config(config_name)
    nlabels = ALL_DATASETS[dataset_name]["n_labels"]
    
    model = UNet(config, n_labels=nlabels).to(device)
    state = torch.load(f"out/{dataset_name}/{config_name}/best_model.pt")["model_state_dict"]
    model.load_state_dict(state)

    dataset_val = SegmentationDataset(dataset_name, "val")
    dataloader_val = DataLoader(dataset_val, shuffle=True)

    labels = ALL_DATASETS[dataset_name]["labels"]
    thresholds = [round(p, 2) for p in np.arange(0.0, 1, 0.01)]
    best_thresholds = []

    for li, label in enumerate(labels):
        scores = {threshold: [] for threshold in thresholds}
        for i, (x, y) in tqdm(enumerate(dataloader_val), f"{dataset_name}/{config_name}", total=max(len(dataloader_val), N)):
            # limit to 100 images to save time
            if i >= N:
                break

            y = y[:, li, :, :]
            y /= 255
            Y_true = y.numpy().squeeze()

            x.requires_grad = True
            x = x.to(device)
            out = model(x)
            out = out[:, li, :, :]
            out = torch.sigmoid(out)
            out = out.detach().cpu().numpy()
            Y_pred = np.squeeze(out)

            for threshold in tqdm(thresholds, "Finding best threshold", 100, False):
                Y_pred_thresholded = (Y_pred >= threshold).astype(int)
                sens = sensitivity(Y_pred_thresholded, Y_true)
                spec = specificity(Y_pred_thresholded, Y_true)
                scores[threshold].append(sens + spec)

        for threshold in scores.keys():
            scores[threshold] = np.mean(scores[threshold])

        best_threshold = max(scores, key=scores.get)
        best_score = scores[best_threshold]

        print(f"Best threshold for {dataset_name}/{config_name}/{label}: {best_threshold} with score {best_score}")
        best_thresholds.append(best_threshold)

    # write to results.json of the corresponding model
    with open(f"out/{dataset_name}/{config_name}/result.json", "r+") as f:
        results = json.load(f)
        results["best_thresholds"] = best_thresholds
        f.seek(0)
        json.dump(results, f)
        f.truncate()


def main(all, dataset):
    if dataset is None:
        assert all
        dataset = "all"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    configurations = os.listdir("configurations")
    configurations.remove("default.json")
    configurations = [config[:-5] for config in configurations if config.endswith(".json")]
    configurations.sort()
    configurations = sorted(configurations, key=len)

    if all:
        datasets = ALL_DATASETS.keys()
    else:
        datasets = [dataset]

    for dataset_name in datasets:
        for config_name in configurations:
            try:
                model_decision_threshold(config_name, dataset_name, device)
            except FileNotFoundError:
                print(f"Could not find files for: {dataset_name}/{config_name}. Skipping...")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--dataset", type=str, help="dataset to preprocess", choices=ALL_DATASETS.keys())
    group.add_argument("-a", "--all", action="store_true", help="preprocess all datasets")
    args = parser.parse_args()
    main(args.all, args.dataset)