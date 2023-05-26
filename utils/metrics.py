import numpy as np
import glob
import os
import torch
import seaborn as sns
from utils.data import SegmentationDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from scipy import signal

def dice_score(arr1, arr2):
    intersection = np.sum(np.logical_and(arr1, arr2))
    union = np.sum(np.logical_or(arr1, arr2))
    if union + intersection == 0:
        return 0
    dice_score = (2.0 * intersection) / (union + intersection)
    return dice_score


def erf_rate_from_dist(erf_dist, trf):
    erf = np.mean(erf_dist, axis=2)
    start = trf[0, 0], trf[0, 1]
    len_x = trf[1, 0] - trf[0, 0] + 1
    len_y = trf[1, 1] - trf[0, 1] + 1
    rf_zone = erf[start[0]:start[0] + len_y, start[1]:start[1]+len_x]

    data = rf_zone.ravel()
    nbins = 100
    hist, bin_edges = np.histogram(data, bins=nbins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    n_range = nbins // 10

    for i, bin in enumerate(hist):
        if bin < np.mean(hist[i-n_range:i+n_range//2]):
            threshold = bin_centers[i]
            erf_rate = np.sum(rf_zone > threshold) / (len_x*len_y) * (1 + rf_zone[rf_zone > threshold].mean())
            return erf_rate


def object_rate(center_trf, mask):
    """Compute the object rate based on a mask and a center trf"""

    len_x = center_trf[1, 0] - center_trf[0, 0] + 1
    len_y = center_trf[1, 1] - center_trf[0, 1] + 1
    trf_area = len_x * len_y

    cols = mask.any(axis=0)
    rows = mask.any(axis=1)

    if not cols.any() or not rows.any():
        return 0.0

    for i, v in enumerate(cols):
        if v:
            start_col = i
            break
    for i, v in enumerate(cols[::-1]):
        if v:
            end_col = len(cols) - i - 1
            break
    for i, v in enumerate(rows):
        if v:
            start_row = i
            break
    for i, v in enumerate(rows[::-1]):
        if v:
            end_row = len(rows) - i - 1 
            break

    object_len_x = end_col - start_col + 1
    object_len_y = end_row - start_row + 1
    object_area = object_len_x * object_len_y
    object_rate = object_area / trf_area

    return object_rate
    

def jaccard_index(im1, im2):
    im1 = im1.astype(bool)
    im2 = im2.astype(bool)

    assert im1.shape == im2.shape

    intersection = np.logical_and(im1, im2)
    union = np.logical_or(im1, im2)
    if union.sum() == 0:
        return 0
    return intersection.sum() / float(union.sum())


def specificity(im1, im2):
    im1 = im1.flatten().astype(bool)
    im2 = im2.flatten().astype(bool)
    cm = confusion_matrix(im1,im2)

    if cm.shape == (1,1):
        return 1

    tn = cm[0,0]
    fp = cm[0,1]
    if tn == 0:
        return 0
    specificity = tn / (tn+fp)
    return specificity


def sensitivity(im1, im2):
    im1 = im1.flatten().astype(bool)
    im2 = im2.flatten().astype(bool)
    cm = confusion_matrix(im1,im2)

    if cm.shape == (1,1):
        return 0

    tp = cm[1,1]
    fn = cm[1,0]
    if tp == 0:
        return 0
    sensitivity = tp / (tp+fn)
    return sensitivity


def accuracy(im1, im2):
    acc = np.sum(im1 == im2) / (576**2)
    return acc
