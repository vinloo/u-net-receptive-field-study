import os
import glob
import shutil
import uuid
import cv2
import random
import argparse
import numpy as np
import nibabel as nib
import pydicom
from PIL import Image
from tqdm import tqdm
from utils.data import ALL_DATASETS


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


def process_mouse_embryo_data(val_rate, test_rate):
    files = glob.glob("data/raw/mouse_embryo_combined/**/*.nii", recursive=True)
    files.sort()

    n_samples = len(files) // 3

    n_train = int(n_samples * (1 - val_rate - test_rate))
    n_val = int(n_samples * val_rate)
    n_test = int(n_samples * test_rate)

    while n_train + n_val + n_test > n_samples:
        n_train -= 1

    for i in tqdm(range(n_samples)):

        if i < n_train:
            target = "train"
        elif i >= n_train and i < n_train + n_val:
            target = "val"
        else:
            target = "test"

        file_img = files[i * 3]
        file_mask_body = files[i * 3 + 1]
        file_mask_bv = files[i * 3 + 2]

        assert file_mask_body.startswith(file_img[:-4]), file_mask_body
        assert file_mask_bv.startswith(file_img[:-4]), file_mask_bv
        assert file_mask_body.endswith("BODY_labels.nii"), file_mask_body
        assert file_mask_bv.endswith("BV_labels.nii"), file_mask_bv

        img = nib.load(file_img).get_fdata()
        img = np.array(img)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)
        mask_body = nib.load(file_mask_body).get_fdata()
        mask_body = np.array(mask_body)
        mask_body = (mask_body - mask_body.min()) / (mask_body.max() - mask_body.min()) * 255
        mask_body = mask_body.astype(np.uint8)
        mask_bv = nib.load(file_mask_bv).get_fdata()
        mask_bv = np.array(mask_bv)
        mask_bv = (mask_bv - mask_bv.min()) / (mask_bv.max() - mask_bv.min()) * 255
        mask_bv = mask_bv.astype(np.uint8)

        assert img.min() >= 0 and img.max() <= 255
        assert set(mask_body.flatten()).issubset({0, 255})
        assert set(mask_bv.flatten()).issubset({0, 255})

        n_slices = img.shape[2]            
        for slice in range(n_slices):
            
            slice_img = Image.fromarray(img[:,:,slice])
            slice_mask_body = Image.fromarray(mask_body[:,:,slice])
            slice_mask_bv = Image.fromarray(mask_bv[:,:,slice])
            

            slice_img = reformat(slice_img)
            slice_mask_body = reformat(slice_mask_body)
            slice_mask_bv = reformat(slice_mask_bv)

            img_id = uuid.uuid4().hex

            os.makedirs(f"data/preprocessed/mouse_embryo/{target}/masks/{img_id}", exist_ok=True)
            slice_img.save(f"data/preprocessed/mouse_embryo/{target}/{img_id}.png")
            slice_mask_body.save(f"data/preprocessed/mouse_embryo/{target}/masks/{img_id}/body.png")
            slice_mask_bv.save(f"data/preprocessed/mouse_embryo/{target}/masks/{img_id}/bv.png")


def process_pancreas_data(val_rate, test_rate):
    files_img = glob.glob("data/raw/pancreas_ct/**/*.dcm", recursive=True)
    files_img.sort()

    files_img_merged = []
    for file in files_img:
        i = int(file.split("/")[6][-4:]) - 1
        if len(files_img_merged) <= i:
            files_img_merged.append([file])
        else:
            files_img_merged[i].append(file)

    files_img = files_img_merged
    files_mask = glob.glob("data/raw/pancreas_ct/**/*.nii.gz", recursive=True)
    files_mask.sort()

    # these are corrputed se we delete them
    del files_img[24], files_img[24], files_img[67], files_img[67]
    del files_mask[24], files_mask[24], files_mask[67], files_mask[67]

    assert len(files_img) == len(files_mask)

    n_samples = len(files_img)
    n_train = int(n_samples * (1 - val_rate - test_rate))
    n_val = int(n_samples * val_rate)
    n_test = int(n_samples * test_rate)

    while n_train + n_val + n_test > n_samples:
        n_train -= 1


    for i, (img_vol, mask_vol) in tqdm(enumerate(zip(files_img, files_mask))):
        masks = nib.load(mask_vol).get_fdata()
        masks = np.array(masks)
        masks = (masks - masks.min()) / (masks.max() - masks.min()) * 255
        masks = masks.astype(np.uint8)

        assert set(masks.flatten()).issubset({0, 255})
        assert len(img_vol) == masks.shape[2], f"Number of images and masks do not match at volume {i}"

        if i < n_train:
            target = "train"
        elif i >= n_train and i < n_train + n_val:
            target = "val"
        else:
            target = "test"

        for j, img in enumerate(img_vol):
            mask = masks[:,:,j]
            img = pydicom.dcmread(img).pixel_array
            if img.max() < 1:
                continue
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)

            
            assert img.min() >= 0 and img.max() <= 255
            assert mask.shape == img.shape

            mask = Image.fromarray(mask)
            img = Image.fromarray(img)
            img = reformat(img)
            mask = reformat(mask)

            img_id = uuid.uuid4().hex

            img.save(f"data/preprocessed/pancreas/{target}/{img_id}.png")
            mask.save(f"data/preprocessed/pancreas/{target}/masks/{img_id}.png")





def preprocess(dataset, val_rate, test_rate):
    clear_data(dataset)

    if dataset == "fetal_head":
        process_fetal_head_data(val_rate, test_rate)
    elif dataset == "breast_cancer":
        process_breast_cancer_data(val_rate, test_rate)
    elif dataset == "mouse_embryo":
        process_mouse_embryo_data(val_rate, test_rate)
    elif dataset == "pancreas":
        process_pancreas_data(val_rate, test_rate)
    else:
        raise ValueError("Invalid dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="dataset to preprocess", choices=ALL_DATASETS.keys(), required=True)
    parser.add_argument("-v", "--val_rate", type=float, default=0.15, help="validation set rate")
    parser.add_argument("-t", "--test_rate", type=float, default=0.15, help="test set rate")
    args = parser.parse_args()

    preprocess(args.dataset, args.val_rate, args.test_rate)


if __name__ == "__main__":
    main()