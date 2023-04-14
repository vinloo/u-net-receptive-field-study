import numpy as np

def dice_score(arr1, arr2):
    intersection = np.sum(np.logical_and(arr1, arr2))
    union = np.sum(np.logical_or(arr1, arr2))
    dice_score = (2.0 * intersection) / (union + intersection)
    return dice_score



