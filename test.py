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
from utils.metrics import dice_score, erf_rate_from_dist, object_rate, jaccard_index, specificity, sensitivity, accuracy
from utils.data import SegmentationDataset
from utils.data import ALL_DATASETS

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def test_model(model, configuration, dataset_name, state, device="cuda"):
    model.eval()
    trf = model.center_trf()

    inputs_test = glob.glob(os.path.join(f"data/preprocessed/{dataset_name}/test", "*.png"))
    masks_test = glob.glob(os.path.join(f"data/preprocessed/{dataset_name}/test/masks", "*.png"))

    dataset_test = SegmentationDataset(inputs_test, masks_test)
    dataloader_test = DataLoader(dataset_test, shuffle=True)

    bt_erf_dist = np.zeros((576, 576, len(dataloader_test)))

    # before_raining erf rate
    for i, (x, y) in tqdm(enumerate(dataloader_test), f"{dataset_name}/{configuration} (BT)", total=len(dataloader_test)):
        y /= 255
        x.requires_grad = True
        x = x.to(device)
        out = model(x)
        
        # ERF rate
        out_center = out[:, :, 288, 288]
        d = torch.autograd.grad(out_center, x)[0]
        d = torch.abs(d)
        d = (d - d.min()) / (d.max() - d.min())
        erf = d.detach().cpu().numpy()
        erf = np.squeeze(erf)
        bt_erf_dist[:, :, i] = erf

    bt_erf_rate = erf_rate_from_dist(bt_erf_dist, trf)

    # metrics after training
    model.load_state_dict(state)

    erf_dist = np.zeros((576, 576, len(dataloader_test)))
    dice_scores = []
    object_rates = []
    accuracies = []
    sensitivities = []
    specificities = []
    jaccard_scores = []

    for i, (x, y) in tqdm(enumerate(dataloader_test), f"{dataset_name}/{configuration}", total=len(dataloader_test)):
        y /= 255
        x.requires_grad = True
        x = x.to(device)
        out = model(x)
        
        # ERF rate
        out_center = out[:, :, 288, 288]
        d = torch.autograd.grad(out_center, x)[0]
        d = torch.abs(d)
        d = (d - d.min()) / (d.max() - d.min())
        d = torch.nan_to_num(d) # if all pixels are predicted 0, then d will be nan
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

    # training time from json file
    with open(f"out/{dataset_name}/{configuration}/result.json", "r") as json_file:
        data = json.load(json_file)
        training_time = data["training_time"]

    return {
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
                state = torch.load(f"out/{dataset_name}/{config_name}/best_model.pt")["model_state_dict"]
                result = test_model(model, config_name, dataset_name, state, device)
                dataset_results[config_name] = pd.Series(result)
            except FileNotFoundError:
                print(f"Could not find files for: {dataset_name}/{config_name}. Skipping...")
        results.append(dataset_results.copy())

    results = pd.concat(results, axis=0, keys=ALL_DATASETS)
    print(results)
    print("\nSaving results to results.csv...")
    results.to_csv("results.csv")