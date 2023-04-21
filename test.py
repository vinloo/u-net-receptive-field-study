import torch
import numpy as np
import glob
import os
from unet import UNet
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.metrics import dice_score, erf_rate_dataset, object_rate
from utils.data import SegmentationDataset

def test_model(model, dataset_name, device="cuda"):
    model.eval()
    
    # model_small_rf_by_hparams.load_state_dict(torch.load(f"results/small_rf_by_hparams/{dataset_name}_best_model.pt")["model_state_dict"])
    inputs_test = glob.glob(os.path.join(f"data/preprocessed/{dataset_name}/test", "*.png"))
    masks_test = glob.glob(os.path.join(f"data/preprocessed/{dataset_name}/test/masks", "*.png"))

    dataset_test = SegmentationDataset(inputs_test, masks_test)
    dataloader_test = DataLoader(dataset_test, shuffle=True)

    trf = model.center_trf()
    erf_dist = np.zeros((576, 576, len(dataloader_test)))
    dice_scores = []
    object_rates = []

    for i, (x, y) in enumerate(dataloader_test):
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
        dicescore = dice_score(out, np.squeeze(y.numpy()))
        dice_scores.append(dicescore)
        obectrate = object_rate(trf, np.squeeze(y.numpy()))
        object_rates.append(obectrate)
    

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

    return {
        "erf_rate": erf_rate,
        "dice_score": dsc,
        "object_rate": obj_rate
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config("default")
    model = UNet(config).to(device)
    model.load_state_dict(torch.load(f"out/fetal_head/default/best_model.pt")["model_state_dict"])
    results = test_model(model, "fetal_head", device)
    print(results)