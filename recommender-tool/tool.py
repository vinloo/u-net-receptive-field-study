import glob
import cv2
import numpy as np
import json
import warnings
import argparse
from dotmap import DotMap
from unet import UNet
from trf import compute_trf
from typing import Literal, Union, Tuple


def __load_config(config: Union[str, dict]) -> DotMap:
    """
    Loads a configuration file or dictionary and returns a DotMap object.

    Parameters:
    config (Union[str, dict]): The configuration file path or dictionary.

    Returns:
    DotMap: A DotMap object containing the configuration parameters.

    Raises:
    AssertionError: If any of the required configuration parameters are missing or have an invalid type.
    """
    if type(config) == str:
        config = json.load(open(config))
    
    config = DotMap(config, _dynamic=False)

    assert config.title is not None, "Title must be specified"
    assert type(config.title) == str, "Title must be a string"
    assert config.depth is not None, "Depth must be specified"
    assert type(config.depth) == int, "Depth must be an integer"
    assert config.attention is not None, "Attention must be specified"
    assert type(config.attention) == bool, "Attention must be a boolean"
    assert config.input_height is not None, "Input height must be specified"
    assert type(config.input_height) == int, "Input height must be an integer"
    assert config.input_width is not None, "Input width must be specified"
    assert type(config.input_width) == int, "Input width must be an integer"
    assert config.input_channels is not None, "Input channels must be specified"
    assert type(config.input_channels) == int, "Input channels must be an integer"
    assert config.output_channels is not None, "Output channels must be specified"
    assert type(config.output_channels) == int, "Output channels must be an integer"
    assert config.encoders_decoders is not None, "Encoders and decoders must be specified"
    assert type(config.encoders_decoders) == list, "Encoders and decoders must be a list"
    assert config.depth > 0, "Depth must be greater than 0"
    assert config.depth == len(config.encoders_decoders), "Depth must be equal to the number of layers"
    assert config.input_height > 0, "Input height must be greater than 0"
    assert config.input_width > 0, "Input width must be greater than 0"
    assert config.input_channels > 0, "Input channels must be greater than 0"
    assert config.output_channels > 0, "Output channels must be greater than 0"

    for i in range(config.depth):
        assert config.encoders_decoders[i] is not None, f"Encoder and decoder at layer {i} must be specified"
        assert type(config.encoders_decoders[i]) == DotMap, f"Encoder and decoder at layer {i} must be a dictionary"
        assert config.encoders_decoders[i].filters is not None, f"Filters at layer {i} must be specified"
        assert type(config.encoders_decoders[i].filters) == int, f"Filters at layer {i} must be an integer"
        assert config.encoders_decoders[i].filters > 0, f"Filters at layer {i} must be greater than 0"
        assert config.encoders_decoders[i].conv_kernel_size is not None, f"Kernel size at layer {i} must be specified"
        assert type(config.encoders_decoders[i].conv_kernel_size) == int, f"Kernel size at layer {i} must be an integer"
        assert config.encoders_decoders[i].conv_kernel_size > 0, f"Kernel size at layer {i} must be greater than 0"
        assert config.encoders_decoders[i].pool_upconv_kernel_size is not None, f"Pool/upconv kernel size at layer {i} must be specified"
        assert type(config.encoders_decoders[i].pool_upconv_kernel_size) == int, f"Pool/upconv kernel size at layer {i} must be an integer"
        assert config.encoders_decoders[i].pool_upconv_kernel_size > 0, f"Pool/upconv kernel size at layer {i} must be greater than 0"
        assert config.encoders_decoders[i].stride is not None, f"Stride at layer {i} must be specified"
        assert type(config.encoders_decoders[i].stride) == int, f"Stride at layer {i} must be an integer"
        assert config.encoders_decoders[i].stride > 0, f"Stride at layer {i} must be greater than 0"

    assert config.bottleneck is not None, "Bottleneck must be specified"
    assert type(config.bottleneck) == DotMap, "Bottleneck must be a dictionary"
    assert config.bottleneck.filters is not None, "Filters at bottleneck must be specified"
    assert type(config.bottleneck.filters) == int, "Filters at bottleneck must be an integer"
    assert config.bottleneck.filters > 0, "Filters at bottleneck must be greater than 0"
    assert config.bottleneck.conv_kernel_size is not None, "Kernel size at bottleneck must be specified"
    assert type(config.bottleneck.conv_kernel_size) == int, "Kernel size at bottleneck must be an integer"
    assert config.bottleneck.stride is not None, "Stride at bottleneck must be specified"
    assert type(config.bottleneck.stride) == int, "Stride at bottleneck must be an integer"
    assert config.bottleneck.stride > 0, "Stride at bottleneck must be greater than 0"
    
    for key in config.keys():
        if key not in ["title", "depth", "attention", "encoders_decoders", "bottleneck", "input_height", "input_width", "input_channels", "output_channels"]:
            warnings.warn(f"Key {key} is not used in the model")
    
    for i, key in enumerate(config.encoders_decoders):
        for key in config.encoders_decoders[i].keys():
            if key not in ["filters", "conv_kernel_size", "pool_upconv_kernel_size", "stride"]:
                warnings.warn(f"Key encoder_decoders[{i}].{key} is not used in the model")

    for key in config.bottleneck.keys():
        if key not in ["filters", "conv_kernel_size", "stride"]:
            warnings.warn(f"Key bottleneck.{key} is not used in the model")

    return config


def mean_RoI_size(masks_dir: str, file_type: str = 'png') -> int:
    """
    Calculates the mean size of the regions of interest (RoIs) in a set of binary masks.

    Parameters:
    masks_dir (str): The directory containing the binary masks.
    file_type (str): The file extension of the binary masks. Default is 'png'.

    Returns:
    int: The mean size of the RoIs in pixels.

    Raises:
    ValueError: If the masks_dir parameter is not a valid directory.
    """

    mask_files = glob.glob(masks_dir + '/*.' + file_type)
    dimensions = []

    for mask_file in mask_files:
        mask = cv2.imread(mask_file, 0)
        mask = mask / 255
        mask = mask.astype(np.uint8)

        # Find the contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get the bounding rectangle of the contour and append the maximum dimension to the list
            rect = cv2.boundingRect(contour)
            height = rect[3]
            width = rect[2]
            dimensions.append(max(height, width))

    return int(np.mean(dimensions))


def image_contrast_level(img: np.ndarray, mask: np.ndarray) -> float:
    RoI = img[mask == 1].flatten()
    not_RoI = img[mask == 0].flatten()

    if len(RoI) == 0 or len(not_RoI) == 0:
        return None

    RoI_hist_norm = np.histogram(RoI, bins=10, range=(0, 256))[
        0] / len(RoI)
    not_RoI_hist_norm = np.histogram(not_RoI, bins=10, range=(0, 256))[
        0] / len(not_RoI)
    contrast = np.mean(np.abs(RoI_hist_norm - not_RoI_hist_norm))
    return contrast



def contrast_segmentable(imgs_dir: str, masks_dir: str, file_type: str = 'png', threshold: float = 0.1) -> Tuple[float, bool]:
    """
    Determines whether the regions of interest (RoIs) in a set of images are contrast segmentable.

    Parameters:
    imgs_dir (str): The directory containing the images.
    masks_dir (str): The directory containing the binary masks.
    file_type (str): The file extension of the images and masks. Default is 'png'.
    threshold (float): The threshold value for contrast. Default is 0.1.

    Returns:
    bool: True if the RoIs are contrast segmentable, False otherwise.
    """

    img_files = glob.glob(imgs_dir + '/*.' + file_type)
    img_files.sort()
    mask_files = glob.glob(masks_dir + '/*.' + file_type)
    mask_files.sort()
    files = list(zip(img_files, mask_files))

    results = []
    contrasts = []

    for img_file, mask_file in files:
        img = cv2.imread(img_file, 0)
        mask = cv2.imread(mask_file, 0)
        mask = mask / 255
        mask = mask.astype(np.uint8)

        contrast = image_contrast_level(img, mask)
        if contrast is None:
            continue
        elif contrast < threshold:
            results.append(False)
        else:
            results.append(True)

        contrasts.append(contrast)

    return np.mean(contrasts), results.count(True) > results.count(False)


def get_current_trf_size(model_config: Union[str, dict], verbose: bool = False) -> dict:
    """
    Returns the maximum TRF size of a UNet model specified by the model_config parameter.

    Parameters:
    model_config (Union[str, dict]): The configuration file or dictionary for the UNet model.
    verbose (bool): Whether to print the output of the compute_trf function. Default is False.

    Returns:
    dict: A dictionary containing the maximum transfer size of the UNet model.

    Raises:
    ValueError: If the model_config parameter is not a valid configuration file or dictionary.
    """
    config = __load_config(model_config)
    model = UNet(config)

    if verbose:
        trf = compute_trf(model, print_output=True)
        return trf[next(reversed(trf))]["max_trf_size"]
    
    return model.max_trf_size()


def recommend_trf(model_config: Union[str, dict], images_dir: str, masks_dir: str, file_type: str = 'png', contrast_threshold: float = 0.1, verbosity: Literal[0, 1, 2] = 0) -> Tuple[int, int, bool]:
    if verbosity >= 2:
        print("Computing...")
        current_trf_size = get_current_trf_size(model_config, verbose=True)
    else:
        current_trf_size = get_current_trf_size(model_config)

    mean_roi_size = mean_RoI_size(masks_dir, file_type)
    contrast, is_contrast_segmentable = contrast_segmentable(images_dir, masks_dir, file_type, contrast_threshold)

    # recommend a TRF size based on the values above
    if is_contrast_segmentable:
        if contrast > contrast_threshold and contrast < 1.5 * contrast_threshold:
            recommend_trf_size = 54
        else:
            recommend_trf_size = mean_roi_size
    else:
        recommend_trf_size = mean_roi_size / (contrast / contrast_threshold)

    recommend_trf_size = int(recommend_trf_size)

    if verbosity >= 1:
        print(f"Current TRF size: {current_trf_size}")
        print(f"Mean RoI size: {mean_roi_size}")
        print(f"Mean contrast level: {contrast:.2f}")
        print(f"Contrast segmentable: {is_contrast_segmentable}")
        print("-" * 30)
        print(f"Recommended TRF size: {recommend_trf_size}")

    return current_trf_size, mean_roi_size, is_contrast_segmentable, recommend_trf_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--model_config", required=True, type=str, help="The configuration file (json) for the U-Net model.")
    parser.add_argument("-i", "--images_dir", required=True, type=str, help="The directory containing the images.")
    parser.add_argument("-m", "--masks_dir", required=True, type=str, help="The directory containing the binary masks.")
    parser.add_argument("-f", "--file_type", required=False, type=str, default='png', help="The file extension of the images and masks. Default is 'png'.")
    parser.add_argument("-t", "--contrast_threshold", required=False, type=float, default=0.1, help="The threshold value for contrast. Default is 0.1.")
    parser.add_argument("-v", "--verbose", required=False, type=bool, default=False, help="Verbose output.")
    args = parser.parse_args()

    verbosity = 2 if args.verbose else 1

    recommend_trf(args.model_config, args.images_dir, args.masks_dir, args.file_type, args.contrast_threshold, verbosity=verbosity)