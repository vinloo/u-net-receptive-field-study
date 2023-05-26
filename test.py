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
from utils.config import load_config, ALL_CONFIGS
from utils.metrics import dice_score, erf_rate_from_dist, object_rate, jaccard_index, specificity, sensitivity, accuracy
from utils.data import SegmentationDataset
from utils.data import ALL_DATASETS

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def test_model(config_name, dataset_name, device="cuda", no_progress=False):
    config = load_config(config_name)
    nlabels = ALL_DATASETS[dataset_name]["n_labels"]
    model = UNet(config, n_labels=nlabels).to(device)

    model.eval()
    trf = model.center_trf()

    dataset_test = SegmentationDataset(dataset_name, "test")
    dataloader_test = DataLoader(dataset_test, shuffle=True)

    labels = ALL_DATASETS[dataset_name]["labels"]
    results = dict()

    for li, label in enumerate(labels):
        model = UNet(config, n_labels=nlabels).to(device)

        bt_erf_dist = np.zeros((576, 576, len(dataloader_test)))

        # before_raining erf rate
        for i, (x, y) in tqdm(enumerate(dataloader_test), f"{dataset_name}/{config_name} (BT)", total=len(dataloader_test), disable=no_progress):
            y = y[:, li, :, :]
            y /= 255
            x.requires_grad = True
            x = x.to(device)
            out = model(x)
            
            # ERF rate
            out_center = out[:, li, 288, 288]
            d = torch.autograd.grad(out_center, x)[0]
            d = torch.abs(d)
            d = (d - d.min()) / (d.max() - d.min())
            d = torch.nan_to_num(d) # if all pixels are predicted 0, then d will be nan
            erf = d.detach().cpu().numpy()
            erf = np.squeeze(erf)
            bt_erf_dist[:, :, i] = erf

        bt_erf_rate = erf_rate_from_dist(bt_erf_dist, trf)

        # metrics after training
        state = torch.load(f"out/{dataset_name}/{config_name}/best_model.pt")["model_state_dict"]
        model.load_state_dict(state)

        erf_dist = np.zeros((576, 576, len(dataloader_test)))
        dice_scores = []
        object_rates = []
        accuracies = []
        sensitivities = []
        specificities = []
        jaccard_scores = []

        # training time from json file
        with open(f"out/{dataset_name}/{config_name}/result.json", "r") as json_file:
            data = json.load(json_file)
            training_time = data["training_time"]

        for i, (x, y) in tqdm(enumerate(dataloader_test), f"{dataset_name}/{config_name}", total=len(dataloader_test), disable=no_progress):
            y = y[:, li, :, :]
            y /= 255
            x.requires_grad = True
            x = x.to(device)
            out = model(x)
            
            # ERF rate
            out_center = out[:, li, 288, 288]
            d = torch.autograd.grad(out_center, x)[0]
            d = torch.abs(d)
            d = (d - d.min()) / (d.max() - d.min())
            d = torch.nan_to_num(d) # if all pixels are predicted 0, then d will be nan
            erf = d.detach().cpu().numpy()
            erf = np.squeeze(erf)
            erf_dist[:, :, i] = erf

            # classification metrics
            out = out[:, li, :, :]
            out = torch.sigmoid(out)
            out = out.detach().cpu().numpy()
            out = (out >= 0.5).astype(int)
            out = np.squeeze(out)

            if y.sum() != 0:
                # dice score
                dicescore = dice_score(out, np.squeeze(y.numpy()))
                dice_scores.append(dicescore)

                # object rate
                obectrate = object_rate(trf, np.squeeze(y.numpy()))
                object_rates.append(obectrate)

                # accuracy
                acc = accuracy(out, np.squeeze(y.numpy()))
                accuracies.append(acc)

                # sensitivity
                sens = sensitivity(out, np.squeeze(y.numpy()))
                sensitivities.append(sens)

                # specificity
                spec = specificity(out, np.squeeze(y.numpy()))
                specificities.append(spec)

                jaccard = jaccard_index(out, np.squeeze(y.numpy()))
                jaccard_scores.append(jaccard)


        # ERF rate
        erf_rate = erf_rate_from_dist(erf_dist, trf)

        # classification metrics
        dsc = np.mean(dice_scores)
        obj_rate = np.mean(object_rates)
        acc = np.mean(accuracies)
        sens = np.mean(sensitivities)
        spec = np.mean(specificities)
        jacc = np.mean(jaccard_scores)

        results[label] = {
            "training_time": training_time,
            "erf_rate_before_training": bt_erf_rate,
            "erf_rate": erf_rate,
            "dice_score": dsc,
            "object_rate": obj_rate,
            "accuracy": acc,
            "sensitivity": sens,
            "specificity": spec,
            "jaccard_score": jacc
        }
    
    return results


def main(all, dataset, config, no_progress=False):
    if dataset is None:
        assert all
        dataset = "all"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if config is None:
        configurations = os.listdir("configurations")
        configurations = [config[:-5] for config in configurations if config.endswith(".json")]
        configurations.sort()
        configurations = sorted(configurations, key=len)
    else:
        configurations = [config]

    results = []

    if all:
        datasets = ALL_DATASETS.keys()
    else:
        datasets = [dataset]

    final_keys = []

    for dataset_name in datasets:
        skip = False
        dataset_results = {label: pd.DataFrame(columns=configurations) for label in ALL_DATASETS[dataset_name]["labels"]}
        for config_name in configurations:
            try:
                result = test_model(config_name, dataset_name, device, no_progress=no_progress)
                for label in result:
                    dataset_results[label][config_name] = pd.Series(result[label])
            except FileNotFoundError:
                print(f"Could not find files for: {dataset_name}/{config_name}. Skipping...")
                skip = True
                continue
        
        if not skip:
            for label in dataset_results:
                final_keys.append(f"{dataset_name}: {label}")
                results.append(dataset_results[label].copy())

    
    ct = datetime.datetime.now()
    ct = ct.strftime("%Y-%m-%d_%H-%M-%S")

    results = pd.concat(results, axis=0, keys=final_keys)

    filename = f"out/test_results_{dataset}_{ct}.csv"
    print(f"Saving results to {filename}")
    results.to_csv(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--dataset", type=str, help="dataset to preprocess", choices=ALL_DATASETS.keys())
    parser.add_argument("-c", "--config", type=str, help="path to config file", choices=ALL_CONFIGS, required=True)
    group.add_argument("-a", "--all", action="store_true", help="preprocess all datasets")
    parser.add_argument("-n", "--no_progress", action="store_true", help="disable progress bar")
    args = parser.parse_args()
    main(args.all, args.dataset, args.config, args.no_progress)