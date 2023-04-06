import os
import glob
import shutil
import uuid
import cv2
import random
from pathlib import Path

def clear_data():
    """Clear existing data in data/preprocessed folder"""
    shutil.rmtree("data/preprocessed")
    os.mkdir("data/preprocessed")
    os.mkdir("data/preprocessed/train")
    os.mkdir("data/preprocessed/val")
    os.mkdir("data/preprocessed/test")
    os.mkdir("data/preprocessed/train/masks")
    os.mkdir("data/preprocessed/val/masks")
    os.mkdir("data/preprocessed/test/masks")
    Path("data/preprocessed/train/.gitkeep").touch()
    Path("data/preprocessed/val/.gitkeep").touch()
    Path("data/preprocessed/test/.gitkeep").touch()
    Path("data/preprocessed/train/masks/.gitkeep").touch()
    Path("data/preprocessed/val/masks/.gitkeep").touch()
    Path("data/preprocessed/test/masks/.gitkeep").touch()


def process_fetal_head_data(val_rate, test_rate):
    files = glob.glob("data/raw/fetal_head/training_set/*")
    files.sort()
    n_samples = len(files) // 2
    n_train = int(n_samples * (1 - val_rate - test_rate))
    n_val = int(n_samples * val_rate)

    sample_ns = list(range(0, n_samples, 2))
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

        # move images to preprocessed folder
        cv2.imwrite(f"data/preprocessed/{target}/masks/{img_id}.png", mask)
        os.system(f"cp {img} data/preprocessed/{target}/{img_id}.png")


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

        os.system(f"cp '{img}' data/preprocessed/{target}/{img_id}.png")
        os.system(f"cp '{ann}' data/preprocessed/{target}/masks/{img_id}.png")
        


def main():
    clear_data()
    # uncomment whichever dataset you want to preprocess
    # process_fetal_head_data(0.2, 0.1)
    process_breast_cancer_data(0.2, 0.1)


if __name__ == '__main__':
    main()