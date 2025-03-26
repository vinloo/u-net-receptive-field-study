# Study of Receptive Field of U-Net and Attention U-Net 
Link to the paper: [https://doi.org/10.1117/1.JMI.11.5.054004](https://doi.org/10.1117/1.JMI.11.5.054004)
## Datasets
### Synthetic Shapes Datasets
> Vincent Loos, Rohit Pardasani, Navchetan Awasthi, Demystifying the effect of receptive field size in U-Net models for medical image segmentation, J. Med. Imag. 11(5), 054004 (2024), doi: 10.1117/1.JMI.11.5.054004.

The preprocessed synthetic datasets with predefined shapes as described in the paper [can be downloaded here](https://github.com/vinloo/u-net-receptive-field-study/releases/tag/v0.1.0) and can be extracted and used directly. 

### Clinical datasets

For copyright reasons the clinical datasets cannot be included into the repository but have to be download and cited separately:

#### Fetal head
> van den Heuvel, T. L. A., de Bruijn, D., de Korte, C. L., and van Ginneken, B. (2018). Automated measurement of fetal head circumference using 2D ultrasound images.

This dataset consists of 2D ultrasound images of fetal heads. It includes 350 training images, 74 validation images, and 76 test images. The images were obtained using a standard clinical ultrasound system, and the fetal head circumference was manually annotated by expert sonographers.

#### Fetal Head 2
> Lu, Y., Zhou, M., Zhi, D., Zhou, M., Jiang, X., Qiu, R., Ou, Z., Wang, H., Qiu, D., Zhong, M., Lu, X., Chen, G., and Bai, J. (2022b). The JNU-IFM dataset for segmenting pubic symphysis-fetal head. Data in Brief, 41:107904.

This is another dataset of 2D ultrasound images of fetal heads, with a larger number of images. It includes 14560 training images, 3240 validation images, and 2875 test images. The images in this dataset were collected from multiple hospitals and were annotated by experienced radiologists.

#### Kidneys
> Daniel, A. J., Buchanan, C. E., Allcock, T., Scerri, D., Cox, E. F., Prestwich, B. L., and Francis, S. T. (2021a). Automated renal segmentation in healthy and chronic kidney disease subjects using a convolutional neural network. Magnetic Resonance in Medicine, 86(2):1125–1136.

> Daniel, A. J., Buchanan, C. E., Allcock, T., Scerri, D., Cox, E. F., Prestwich, B. L., and Francis, S. T. (2021b). T2-weighted kidney mri segmentation.

This dataset consists of 3D MRI images of kidneys \citep{kidney.1, kidney.2}. It includes 454 training images, 91 validation images, and 104 test images. The images were acquired using a 3T MRI scanner and the kidney regions were manually segmented by radiologists.

#### Lungs
> Kassamali, R. H. and Jafarieh, S. (2014). Passion and hard work produces high quality research in uk: response to focus on china: should clinicians engage in research? and lessons from other countries. Quantitative Imaging in Medicine and Surgery,
4(6).

> Viacheslav Danilov (2022). Chest x-ray dataset for lung segmentation.

This dataset consists of 2D X-Ray images of lungs. It includes 396 training images, 84 validation images, and 86 test images. The images were collected from a variety of patients with different lung conditions, providing a diverse dataset for training and testing.

#### Thyroid
> Wunderling, T., Golla, B., Poudel, P., Arens, C., Friebe, M., and Hansen, C. (2017). Comparison of thyroid segmentation techniques for 3d ultrasound. In Proceedings of SPIE Medical Imaging, Orlando, USA.

This dataset consists of 3D ultrasound images of the thyroid \citep{thyroid}. It includes 3160 training images, 439 validation images, and 510 test images. The images were acquired using a high-frequency linear array transducer and the thyroid regions were manually segmented by experienced clinicians.

#### Nerve
> Anna Montoya, W. C. (2016). Ultrasound nerve segmentation.

This dataset consists of 2D ultrasound images of nerves \citep{nerve}. It includes 1610 training images, 364 validation images, and 349 test images. The images were collected from a variety of patients and the nerve structures were manually annotated by expert radiologists.

## Preprocessing
Unzip the datasets in the `preprocessed/raw` folder and run `python preprocess.py` with the following arguments.
> NOTE: the preprocessing script does not take into account class balancing. This has to be inspected and corrected seperately. An example on how to do this is shown in [`notebooks/dataset_balance.ipynb`](notebooks/dataset_balance.ipynb)

| **Argument**         |          **Flag**         | **Required** | **Type** |       **Default value**       |
|----------------------|:-------------------------:|:------------:|:--------:|:-----------------------------:|
| Dataset              | `-d` or `--dataset`       |      yes     |    str   |                          None |
| Validation set rate  | `-v` or `--val_rate`      |      no      |   float  |                          0.15 |
| Test set rate        | `-t` or `--test_rate`     |      no      |   float  |                          0.15 |
| Augmentation factor  | `-a` or `--augmentations` |      no      |    int   |                             0 |


## Training
Training the model can be done by running `python train.py -d <dataset> -c <configuration>`. The dataset and configuration are the only required argument, all other arguments have default values as listed in this table:
| **Argument**      |          **Flag**         | **Required** | **Type** |       **Default value**       |
|-------------------|:-------------------------:|:------------:|:--------:|:-----------------------------:|
| Dataset           | `-d` or `--dataset`       |      yes     |    str   |                          None |
| Model config file | `-c` or `--config`        |      no      |    str   |                          None |
| Epochs            | `-e` or `--epochs`        |      no      |    int   |                           200 |
| Batch size        | `-b` or `--batch_size`    |      no      |    int   |                             2 |
| Learning rate     | `-l` or `--learning_rate` |      no      |   float  |                        0.0001 |
| Output folder     | `-o` or `--output_dir`    |      no      |    str   |                         "out" |
| Hide progress bar | `-n` or `--no_progress`   |      no      |          |                         False |
| Attention U-Net   | `-a` or `--attention`     |      no      |          |                         False |

After training, the best model for each dataset and configuration is stored in `out/{dataset_name}/{config_name}`.

## Testing
Testing the model can be done by running `python test.py -d <dataset> -c <configuration>`. The dataset/all and configuration are the only required argument, all other arguments have default values as listed in this table:
| **Argument**      |          **Flag**         | **Required** | **Type** |       **Default value**       |
|-------------------|:-------------------------:|:------------:|:--------:|:-----------------------------:|
| Dataset           | `-d` or `--dataset`       |      yes     |    str   |                          None |
| All datasets      | `--all`                   |              |          |                         False |
| Model config file | `-c` or `--config`        |      no      |    str   |                          None |
| Hide progress bar | `-n` or `--no_progress`   |      no      |          |                         False |
| Attention U-Net   | `-a` or `--attention`     |      no      |          |                         False |

# Tool: U-Net Receptive Field Size Recommender
The tool in [`recommender-tool/`](recommender-tool) utilizes a heuristic approach which will most likely not result in the optimal RF size, but it will recommend a good starting point based on the findings of the study.

## How to Use
The tool can be used in two ways: as a CLI tool or as a Python module. Either way, it requires three inputs:
 * A JSON file which contains the architectural layout of the U-Net or Attetnion U-Net (Example available in [`recommender-tool/example/u-net-configuration.json`](recommender-tool/example/u-net-configuration.json)).
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

## Citing this work
> Vincent Loos, Rohit Pardasani, Navchetan Awasthi, Demystifying the effect of receptive field size in U-Net models for medical image segmentation, J. Med. Imag. 11(5), 054004 (2024), doi: 10.1117/1.JMI.11.5.054004.
