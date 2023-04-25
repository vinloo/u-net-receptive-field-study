import torch
import numpy as np
import glob
import os
import pandas as pd
import json
from tqdm import tqdm
from unet import UNet
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.metrics import dice_score, erf_rate_dataset, object_rate
from utils.data import SegmentationDataset
from preprocess_data import ALL_DATASETS

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def test_model(model, configuration, dataset_name, device="cuda"):
    model.eval()
    
    inputs_test = glob.glob(os.path.join(f"data/preprocessed/{dataset_name}/test", "*.png"))
    masks_test = glob.glob(os.path.join(f"data/preprocessed/{dataset_name}/test/masks", "*.png"))

    dataset_test = SegmentationDataset(inputs_test, masks_test)
    dataloader_test = DataLoader(dataset_test, shuffle=True)

    trf = model.center_trf()
    erf_dist = np.zeros((576, 576, len(dataloader_test)))
    dice_scores = []
    object_rates = []
    accuratcies = []
    sensitivities = []
    specificities = []
    jaccard_scores = []

    for i, (x, y) in tqdm(enumerate(dataloader_test), f"{dataset_name}/{configuration}", total=len(dataloader_test)):
        x.requires_grad = True
        x = x.to(device)
        y_true = y / 255
        out = model(x)
        
        # ERF rate
        out_center = out[:, :, 288, 288]
        d = torch.autograd.grad(out_center, x)[0]
        d = torch.abs(d)
        d = (d - d.min()) / (d.max() - d.min())
        erf = d.detach().cpu().numpy()
        erf = np.squeeze(erf)
        erf_dist[:, :, i] = erf

        # classification metrics
        out = torch.sigmoid(out)
        out = out.detach().cpu().numpy()
        out = (out >= 0.5).astype(int)
        out = np.squeeze(out)

        # dice score
        dicescore = dice_score(out, np.squeeze(y.numpy()))
        dice_scores.append(dicescore)
        obectrate = object_rate(trf, np.squeeze(y.numpy()))
        object_rates.append(obectrate)

        # accuracy
        accuracy = np.sum(out == np.squeeze(y.numpy())) / (576*576)
        accuratcies.append(accuracy)

        # sensitivity
        numerator = np.sum((out == 1) & (np.squeeze(y.numpy()) == 1))
        denominator = np.sum(np.squeeze(y.numpy()) == 1)
        if denominator == 0:
            sensitivity = 0
        else:
            sensitivity = numerator / denominator
        sensitivities.append(sensitivity)

        # specificity
        specificity = np.sum((out == 0) & (np.squeeze(y.numpy()) == 0)) / np.sum(np.squeeze(y.numpy()) == 0)
        specificities.append(specificity)

        # jaccard score
        numerator = np.sum((out == 1) & (np.squeeze(y.numpy()) == 1))
        denominator = np.sum((out == 1) | (np.squeeze(y.numpy()) == 1))
        if denominator == 0:
            jaccard = 0
        else:
            jaccard = numerator / denominator
        jaccard_scores.append(jaccard)


    # ERF rate
    erf = np.mean(erf_dist, axis=2)
    start = trf[0, 0], trf[0, 1]
    len_x = trf[1, 0] - trf[0, 0] + 1
    len_y = trf[1, 1] - trf[0, 1] + 1
    rf_zone = erf[start[0]:start[0] + len_y, start[1]:start[1]+len_x]
    erf_rate = np.sum(rf_zone > 0.025) / (len_x*len_y) * (1 + rf_zone[rf_zone > 0.025].mean())

    # classification metrics
    dsc = np.mean(dice_scores)
    obj_rate = np.mean(object_rates)
    acc = np.mean(accuratcies)
    sens = np.mean(sensitivities)
    spec = np.mean(specificities)
    jacc = np.mean(jaccard_scores)

    # training time from json file
    with open(f"out/{dataset_name}/{configuration}/result.json", "r") as json_file:
        data = json.load(json_file)
        training_time = data["training_time"]

    return {
        "training_time": training_time,
        "erf_rate": erf_rate,
        "dice_score": dsc,
        "object_rate": obj_rate,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "jaccard_score": jacc
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    configurations = os.listdir("configurations")
    configurations.remove("default.json")
    configurations = [config[:-5] for config in configurations if config.endswith(".json")]
    configurations.sort()
    configurations = sorted(configurations, key=len)

    results = []

    for dataset_name in ALL_DATASETS:
        dataset_results = pd.DataFrame(columns=configurations)
        for config_name in configurations:
            try:
                config = load_config(config_name)
                model = UNet(config).to(device)
                model.load_state_dict(torch.load(f"out/{dataset_name}/{config_name}/best_model.pt")["model_state_dict"])
                result = test_model(model, config_name, dataset_name, device)
                dataset_results[config_name] = pd.Series(result)
            except FileNotFoundError:
                print(f"Could not find files for: {dataset_name}/{config_name}. Skipping...")
        results.append(dataset_results.copy())

    results = pd.concat(results, axis=0, keys=ALL_DATASETS)
    print(results)
    print("\nSaving results to results.csv...")
    results.to_csv("results.csv")