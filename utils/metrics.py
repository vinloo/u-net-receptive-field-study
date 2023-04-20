import numpy as np
import glob
import os
import torch
from utils.data import SegmentationDataset
from torch.utils.data import DataLoader

def dice_score(arr1, arr2):
    intersection = np.sum(np.logical_and(arr1, arr2))
    union = np.sum(np.logical_or(arr1, arr2))
    dice_score = (2.0 * intersection) / (union + intersection)
    return dice_score


def batch_erf_rate(batch_in, out, trf):
    """Compute the ERF rate for a batch of images, to be used by trainer"""
    erf_rates = []
    for i in range(batch_in.shape[0]):
        out_center = out[i, :, 288, 288]
        d = torch.autograd.grad(out_center, batch_in, retain_graph=True)[0]
        img = d[i, :, :, :]
        img = torch.abs(img)
        img = (img - img.min()) / (img.max() - img.min())

        img = torch.squeeze(img)

        start = trf[0, 0], trf[0, 1]
        len_x = trf[1, 0] - trf[0, 0] + 1
        len_y = trf[1, 1] - trf[0, 1] + 1

        rf_zone = img[start[0]:start[0] + len_y, start[1]:start[1]+len_x]
        erf_rate = torch.sum(rf_zone > 0.025) / (len_x*len_y) * (1 + rf_zone[rf_zone > 0.025].mean())
        erf_rates.append(erf_rate.item())

    return erf_rates


def erf_rate_dataset(model, dataset_name, device="cuda"):
    """Compute the ERF rate based on an entire dataset for a given model"""
    trf = model.center_trf()
    inputs = glob.glob(os.path.join(f"data/preprocessed/{dataset_name}/*", "*.png"))
    masks = glob.glob(os.path.join(f"data/preprocessed/{dataset_name}/*/masks", "*.png"))
    dataset_test = SegmentationDataset(inputs, masks)
    dataloader_test = DataLoader(dataset_test, shuffle=True)

    dist = np.zeros((576, 576, len(dataloader_test)))

    for i, (x, _) in enumerate(dataloader_test):
        x.requires_grad = True
        x = x.to(device)
        out = model(x)
        out_center = out[:, :, 288, 288]

        d = torch.autograd.grad(out_center, x)[0]
        d = torch.abs(d)
        d = (d - d.min()) / (d.max() - d.min())

        img = d.detach().cpu().numpy()
        img = np.squeeze(img)

        dist[:, :, i] = img
    
    img = np.mean(dist, axis=2)

    start = trf[0, 0], trf[0, 1]
    len_x = trf[1, 0] - trf[0, 0] + 1
    len_y = trf[1, 1] - trf[0, 1] + 1

    rf_zone = img[start[0]:start[0] + len_y, start[1]:start[1]+len_x]
    erf_rate = np.sum(rf_zone > 0.025) / (len_x*len_y) * (1 + rf_zone[rf_zone > 0.025].mean())
    return erf_rate



