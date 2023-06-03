import os
import glob
import shutil
import uuid
import random
import argparse
import nibabel as nib
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from utils.data import ALL_DATASETS


def random_augmentations(img: Image, mask: Image):
    """
    Applies random augmentations to an image and its corresponding mask.

    Args:
        img (PIL.Image): The input image.
        mask (PIL.Image): The corresponding mask.

    Returns:
        tuple: A tuple containing the augmented image and mask.

    """
    p_hflip = random.random()
    p_vflip = random.random()
    p_rot = random.random()

    if p_hflip > 0.5:
        img = transforms.functional.hflip(img)
        mask = transforms.functional.hflip(mask)

    if p_vflip > 0.5:
        img = transforms.functional.vflip(img)
        mask = transforms.functional.vflip(mask)

    if p_rot > 0.5:
        angle = random.choice([90, 180, 270])
        img = transforms.functional.rotate(img, angle)
        mask = transforms.functional.rotate(mask, angle)

    return img, mask


def reformat(img: Image) -> Image:
    """
    Converts an image to grayscale and resizes it to a fixed size.

    Args:
        img (PIL.Image): The input image.

    Returns:
        PIL.Image: The reformatted image.

    """
    img = img.convert('L')
    img = img.resize((576, 576))
    return img


def clear_data(dataset: str):
    """
    Clears existing data in the 'data/preprocessed' folder for a given dataset.

    Args:
        dataset (str): The name of the dataset to clear.

    """
    # Remove the existing dataset folder if it exists
    if os.path.exists(f"data/preprocessed/{dataset}"):
        shutil.rmtree(f"data/preprocessed/{dataset}")

    # Create new directories for the dataset and its subdirectories
    os.mkdir(f"data/preprocessed/{dataset}")
    os.mkdir(f"data/preprocessed/{dataset}/train")
    os.mkdir(f"data/preprocessed/{dataset}/val")
    os.mkdir(f"data/preprocessed/{dataset}/test")
    os.mkdir(f"data/preprocessed/{dataset}/train/masks")
    os.mkdir(f"data/preprocessed/{dataset}/val/masks")
    os.mkdir(f"data/preprocessed/{dataset}/test/masks")


def example_2D_preprocessing_function(val_rate, test_rate, n_augmentations) -> None:
    """
    Preprocesses 2D images and masks for the example dataset.

    Args:
        val_rate (float): The validation set ratio.
        test_rate (float): The test set ratio.
        n_augmentations (int): The number of augmentations to apply to each image.

    Returns:
        None

    """
    # Get the file paths for the images and masks
    img_files = glob.glob("data/raw/example_2D/images/*.png")
    img_files.sort()
    mask_files = glob.glob("data/raw/example_2D/masks/*.png")
    mask_files.sort()

    # Shuffle the pairs of images and masks
    pairs = list(zip(img_files, mask_files))
    random.shuffle(pairs)

    # Split the data into train, validation, and test sets
    n_samples = len(pairs)
    n_val = int(n_samples * val_rate)
    n_test = int(n_samples * test_rate)
    n_train = n_samples - n_val - n_test

    # Preprocess each image and mask and save them to the appropriate directory
    for i, (img_file, mask_file) in tqdm(enumerate(pairs)):
        assert os.path.basename(img_file) == os.path.basename(mask_file)

        if i < n_train:
            target = "train"
        elif i < n_train + n_val:
            target = "val"
        else:
            target = "test"

        img = Image.open(img_file)
        img = reformat(img)
        mask = Image.open(mask_file)
        mask = reformat(mask)

        img_id = uuid.uuid4().hex

        img.save(f"data/preprocessed/example_2D/{target}/{img_id}.png")
        mask.save(f"data/preprocessed/example_2D/{target}/masks/{img_id}.png")

        for _ in range(n_augmentations):
            img_id = uuid.uuid4().hex
            img_aug, mask_aug = random_augmentations(img, mask)
            img_aug.save(f"data/preprocessed/example_2D/{target}/{img_id}.png")
            mask_aug.save(
                f"data/preprocessed/example_2D/{target}/masks/{img_id}.png")


def example_3D_preprocessing_function(val_rate, test_rate, n_augmentations) -> None:
    """
    Preprocesses 3D images and masks for the example dataset.

    Args:
        val_rate (float): The validation set ratio.
        test_rate (float): The test set ratio.
        n_augmentations (int): The number of augmentations to apply to each image.

    Returns:
        None

    """
    # Get the file paths for the images and masks
    img_files = glob.glob("data/raw/example_3D/images/*.nii.gz")
    img_files.sort()
    mask_files = glob.glob("data/raw/example_3D/masks/*.nii.gz")
    mask_files.sort()

    # Shuffle the pairs of images and masks
    pairs = list(zip(img_files, mask_files))
    random.shuffle(pairs)

    # Split the data into train, validation, and test sets
    n_samples = len(pairs)
    n_val = int(n_samples * val_rate)
    n_test = int(n_samples * test_rate)
    n_train = n_samples - n_val - n_test

    # Preprocess each image and mask and save them to the appropriate directory
    for i, (img_file, mask_file) in tqdm(enumerate(pairs)):
        assert os.path.basename(img_file) == os.path.basename(mask_file)

        if i < n_train:
            target = "train"
        elif i < n_train + n_val:
            target = "val"
        else:
            target = "test"

        img = nib.load(img_file).get_fdata()
        mask = nib.load(mask_file).get_fdata()

        assert img.shape == mask.shape

        for j in range(img.shape[2]):
            img_slice = Image.fromarray(img[:, :, j])
            img_slice = reformat(img_slice)
            mask_slice = Image.fromarray(mask[:, :, j])
            mask_slice = reformat(mask_slice)

            img_id = uuid.uuid4().hex

            img_slice.save(
                f"data/preprocessed/example_3D/{target}/{img_id}.png")
            mask_slice.save(
                f"data/preprocessed/example_3D/{target}/masks/{img_id}.png")

            for _ in range(n_augmentations):
                img_id = uuid.uuid4().hex
                img_aug, mask_aug = random_augmentations(img_slice, mask_slice)
                img_aug.save(
                    f"data/preprocessed/example_3D/{target}/{img_id}.png")
                mask_aug.save(
                    f"data/preprocessed/example_3D/{target}/masks/{img_id}.png")


def preprocess(dataset, val_rate, test_rate, n_augmentations=0) -> None:
    """
    Preprocesses images and masks for a given dataset.

    Args:
        dataset (str): The name of the dataset to preprocess.
        val_rate (float): The validation set ratio.
        test_rate (float): The test set ratio.
        n_augmentations (int): The number of augmentations to apply to each image.

    Returns:
        None

    Raises:
        ValueError: If an invalid dataset name is provided.

    """
    clear_data(dataset)

    # all dataset-specific preprocessing functions
    if dataset == "example_2D":
        example_2D_preprocessing_function(val_rate, test_rate, n_augmentations)
    elif dataset == "example_3D":
        example_3D_preprocessing_function(val_rate, test_rate, n_augmentations)
    # elif dataset == "another_dataset":
    #     preprocess_another_dataset(val_rate, test_rate, n_augmentations)
    else:
        raise ValueError("Invalid dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="dataset to preprocess",
                        choices=ALL_DATASETS.keys(), required=True)
    parser.add_argument("-a", "--augmentations", type=int,
                        default=0, help="number of augmentations to perform")
    parser.add_argument("-v", "--val_rate", type=float,
                        default=0.15, help="validation set rate")
    parser.add_argument("-t", "--test_rate", type=float,
                        default=0.15, help="test set rate")
    args = parser.parse_args()

    preprocess(args.dataset, args.val_rate, args.test_rate, args.augmentations)


if __name__ == "__main__":
    main()
