import os
import glob
import shutil
import uuid
import cv2
import random
import argparse
import numpy as np
import nibabel as nib
from PIL import Image


ALL_DATASETS = ["fetal_head", "breast_cancer", "mouse_embryo_body", "mouse_embryo_bv"]


def reformat(img):
    # convert to grayscale and scale to fixed size
    img = img.convert('L')
    img = img.resize((576, 576))
    return img


def clear_data(dataset):
    """Clear existing data in data/preprocessed folder"""

    if dataset == "mouse_embryo":
        clear_data("mouse_embryo_body")
        clear_data("mouse_embryo_bv")
        return

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

    body_imgs = np.empty((576, 576, 0))
    body_masks = np.empty((576, 576, 0))
    bv_imgs = np.empty((576, 576, 0))
    bv_masks = np.empty((576, 576, 0))

    for i in range(len(files) // 3):
        file_img = files[i * 3]
        file_mask_body = files[i * 3 + 1]
        file_mask_bv = files[i * 3 + 2]

        assert file_mask_body.startswith(file_img[:-4]), file_mask_body
        assert file_mask_bv.startswith(file_img[:-4]), file_mask_bv
        assert file_mask_body.endswith("BODY_labels.nii"), file_mask_body
        assert file_mask_bv.endswith("BV_labels.nii"), file_mask_bv

        img = nib.load(file_img).get_fdata()
        img = np.array(img) * 255
        mask_body = nib.load(file_mask_body).get_fdata()
        mask_body = np.array(mask_body)
        mask_bv = nib.load(file_mask_bv).get_fdata()
        mask_bv = np.array(mask_bv)

        n_slices = img.shape[2]

        # take some slices from the image, add more to increase dataset size
        slices = [n_slices // 4, n_slices // 3, n_slices // 2, n_slices // 3 * 2, n_slices // 4 * 3]

        for slice in slices:
            slice = n_slices // 2
            slice_img = Image.fromarray(img[:,:,slice])
            slice_mask_body = Image.fromarray(mask_body[:,:,slice])
            slice_mask_bv = Image.fromarray(mask_bv[:,:,slice])

            slice_img = reformat(slice_img)
            slice_img = np.array(slice_img) / 255
            slice_mask_body = reformat(slice_mask_body)
            slice_mask_bv = reformat(slice_mask_bv)

            if np.sum(slice_mask_body) > 0:
                body_imgs = np.dstack((body_imgs, slice_img))
                body_masks = np.dstack((body_masks, np.array(slice_mask_body)))

            if np.sum(slice_mask_bv) > 0:
                bv_imgs = np.dstack((bv_imgs, slice_img))
                bv_masks = np.dstack((bv_masks, np.array(slice_mask_bv)))

    assert body_imgs.shape == body_masks.shape
    assert bv_imgs.shape == bv_masks.shape

    body_idxs = np.arange(body_imgs.shape[2])
    np.random.shuffle(body_idxs)
    bv_idxs = np.arange(bv_imgs.shape[2])
    np.random.shuffle(bv_idxs)

    body_train_idxs = body_idxs[:int(body_imgs.shape[2] * (1 - val_rate - test_rate))]
    body_val_idxs = body_idxs[int(body_imgs.shape[2] * (1 - val_rate - test_rate)):int(body_imgs.shape[2] * (1 - test_rate))]
    body_test_idxs = body_idxs[int(body_imgs.shape[2] * (1 - test_rate)):]

    bv_train_idxs = bv_idxs[:int(bv_imgs.shape[2] * (1 - val_rate - test_rate))]
    bv_val_idxs = bv_idxs[int(bv_imgs.shape[2] * (1 - val_rate - test_rate)):int(bv_imgs.shape[2] * (1 - test_rate))]
    bv_test_idxs = bv_idxs[int(bv_imgs.shape[2] * (1 - test_rate)):]

    for i in body_idxs:
        img = Image.fromarray((body_imgs[:,:,i] * 255).astype(np.uint8))
        mask = Image.fromarray((body_masks[:,:,i] * 255).astype(np.uint8))

        if i in body_train_idxs:
            target = "train"
        elif i in body_val_idxs:
            target = "val"
        elif i in body_test_idxs:
            target = "test"

        img_id = uuid.uuid4().hex

        img.save(f"data/preprocessed/mouse_embryo_body/{target}/{img_id}.png")
        mask.save(f"data/preprocessed/mouse_embryo_body/{target}/masks/{img_id}.png")

    for i in bv_idxs:
        img = Image.fromarray((bv_imgs[:,:,i] * 255).astype(np.uint8))
        mask = Image.fromarray((bv_masks[:,:,i] * 255).astype(np.uint8))

        if i in bv_train_idxs:
            target = "train"
        elif i in bv_val_idxs:
            target = "val"
        elif i in bv_test_idxs:
            target = "test"

        img_id = uuid.uuid4().hex

        img.save(f"data/preprocessed/mouse_embryo_bv/{target}/{img_id}.png")
        mask.save(f"data/preprocessed/mouse_embryo_bv/{target}/masks/{img_id}.png")


        


def preprocess(dataset, val_rate, test_rate):
    clear_data(dataset)

    if dataset == "fetal_head":
        process_fetal_head_data(val_rate, test_rate)
    elif dataset == "breast_cancer":
        process_breast_cancer_data(val_rate, test_rate)
    elif dataset == "mouse_embryo":
        process_mouse_embryo_data(val_rate, test_rate)
    else:
        raise ValueError("Invalid dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="dataset to preprocess", choices=ALL_DATASETS, required=True)
    parser.add_argument("-v", "--val_rate", type=float, default=0.15, help="validation set rate")
    parser.add_argument("-t", "--test_rate", type=float, default=0.15, help="test set rate")
    args = parser.parse_args()

    preprocess(args.dataset, args.val_rate, args.test_rate)


if __name__ == "__main__":
    main()