# U-Net Receptive Field Size Recommender
This tool utilizes a heuristic approach which will most likely not result in the optimal RF size, but it will recommend a good starting point based on the findings of the study.

## How to Use
The tool can be used in two ways: as a CLI tool or as a Python module. Either way, it requires three inputs:
 * A JSON file which contains the architectural layout of the U-Net or Attetnion U-Net (Example available in [`example/u-net-configuration.json`](example/u-net-configuration.json)).
 * The path of the directory that contains the dataset's images
 * The path of the directory that contains the dataset's masks

### As a CLI tool
Run `python tool.py` with the following arguments:
| Argument Short Form | Argument Long Form | Required | Type | Default Value | Description |
| ------------------- | ------------------ | -------- | ---- | ------------- | ----------- |
| -c | --model_config | Yes | str | None | The configuration file (json) for the U-Net model. |
| -i | --images_dir | Yes | str | None | The directory containing the images. |
| -m | --masks_dir | Yes | str | None | The directory containing the binary masks. |
| -f | --file_type | No | str | 'png' | The file extension of the images and masks. Default is 'png'. |
| -t | --contrast_threshold | No | float | 0.1 | The threshold value for contrast. Default is 0.1. |

### As a Python module
**Coming Soon!**


## Citing this work
> **TODO** Include citation
