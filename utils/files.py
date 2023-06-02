import os
import shutil

def get_output_path(dataset_name, config_name, out_dir, attention, clear_existing=False):
    """Create output path for the model and the results"""
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