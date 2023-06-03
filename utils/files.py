import os
import shutil


def get_output_path(dataset_name: str, config_name: str, out_dir: str, attention: bool, clear_existing: bool = False) -> str:
    """
    Create output path for the model and the results.

    Args:
        dataset_name (str): The name of the dataset.
        config_name (str): The name of the configuration.
        out_dir (str): The output directory.
        attention (bool): Whether to use attention.
        clear_existing (bool, optional): Whether to clear existing output directory. Defaults to False.

    Returns:
        str: The output path.

    """
    if attention:
        path = f"{out_dir}/attention_{dataset_name}/{config_name}"
    else:
        path = f"{out_dir}/{dataset_name}/{config_name}"
    if not os.path.exists(path):
        os.makedirs(path)
    elif clear_existing:
        shutil.rmtree(path)
        os.makedirs(path)
    return path
