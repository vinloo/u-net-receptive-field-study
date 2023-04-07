import os
import glob
import shutil
import uuid
import cv2
import random
import argparse
from PIL import Image


ALL_DATASETS = ["fetal_head", "breast_cancer"]


def reformat(img):
    # convert to grayscale and scale to fixed size
    img = img.convert('L')
    img = img.resize((576, 576))
    return img


def clear_data(dataset):
    """Clear existing data in data/preprocessed folder"""
    if os.path.exists(f"data/preprocessed/{dataset}"):
        shutil.rmtree(f"data/preprocessed/{dataset}")
    os.mkdir(f"data/preprocessed/{dataset}")
    os.mkdir(f"data/preprocessed/{dataset}/train")
    os.mkdir(f"data/preprocessed/{dataset}/val")
    os.mkdir(f"data/preprocessed/{dataset}/test")
    os.mkdir(f"data/preprocessed/{dataset}/train/masks")
    os.mkdir(f"data/preprocessed/{dataset}/val/masks")
    os.mkdir(f"data/preprocessed/{dataset}/test/masks")


def process_fetal_head_data(val_rate, test_rate):
    files = glob.glob("data/raw/fetal_head/training_set/*")
    files.sort()
    n_samples = len(files) // 2
    n_train = int(n_samples * (1 - val_rate - test_rate))
    n_val = int(n_samples * val_rate)

    sample_ns = list(range(0, n_samples, 2))
    random.seed("split seed") # to make sure it is split the same way every time
    random.shuffle(sample_ns)

    for i in sample_ns:
        img = files[i]
        ann = files[i + 1]
        assert ann.endswith("_Annotation.png") and not img.endswith("_Annotation.png")

        img_id = uuid.uuid4().hex

        # split data into train, val, test
        if i < n_train:
            target = "train"
        elif i < n_train + n_val:
            target = "val"
        else:
            target = "test"

        # fill the annotation's mask with white pixels
        mask = cv2.imread(ann, cv2.IMREAD_GRAYSCALE)
        thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cv2.fillPoly(mask, cnts, [255,255,255])

        ann = Image.fromarray(mask)
        img = Image.open(img)
        img = reformat(img)
        ann = reformat(ann)

        img.save(f"data/preprocessed/fetal_head/{target}/{img_id}.png")
        ann.save(f"data/preprocessed/fetal_head/{target}/masks/{img_id}.png")


def process_breast_cancer_data(val_rate, test_rate):
    files = glob.glob("data/raw/breast_cancer/Dataset_BUSI_with_GT/*/*")
    files.sort()

    # temporary solution: ignore images with more than 1 mask
    multimask_ns = {f[f.find("(")+1:f.find(")")] for f in files if f.endswith("_mask_1.png")}
    files = [f for f in files if f[f.find("(")+1:f.find(")")] not in multimask_ns]

    n_samples = len(files) // 2
    n_train = int(n_samples * (1 - val_rate - test_rate))
    n_val = int(n_samples * val_rate)

    sample_ns = list(range(0, n_samples, 2))
    random.seed("split seed") # to make sure it is split the same way every time
    random.shuffle(sample_ns)
    
    for i in sample_ns:
        img = files[i]
        ann = files[i + 1]
        assert ann.endswith("_mask.png") and not img.endswith("_mask.png")

        img_id = uuid.uuid4().hex

        # split data into train, val, test
        if i < n_train:
            target = "train"
        elif i < n_train + n_val:
            target = "val"
        else:
            target = "test"

        img = Image.open(img)
        ann = Image.open(ann)
        img = reformat(img)
        ann = reformat(ann)

        img.save(f"data/preprocessed/breast_cancer/{target}/{img_id}.png")
        ann.save(f"data/preprocessed/breast_cancer/{target}/masks/{img_id}.png")
        


def preprocess(dataset, val_rate, test_rate):
    clear_data(dataset)

    if dataset == "fetal_head":
        process_fetal_head_data(val_rate, test_rate)
    elif dataset == "breast_cancer":
        process_breast_cancer_data(val_rate, test_rate)
    else:
        raise ValueError("Invalid dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="dataset to preprocess", choices=ALL_DATASETS, required=True)
    parser.add_argument("-v", "--val_rate", type=float, default=0.2, help="validation set rate")
    parser.add_argument("-t", "--test_rate", type=float, default=0.1, help="test set rate")
    args = parser.parse_args()

    preprocess(args.dataset, args.val_rate, args.test_rate)


if __name__ == "__main__":
    main()