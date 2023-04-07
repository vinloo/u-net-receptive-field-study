# Study of Receptive Field of U-Net and Attention U-Net 

The U-Net in this code consists of 4 encoder blocks, a bottleneck, and 4 decoder blocks. All parameters for all layers in all blocks can be easily tuned by creating a configuration file and passing it into the UNet class. I have already created a [default configuration file](configurations/default.json) to test the UNet. This is very useful as the receptive field depends on the kernel size, stride and padding of each layer; so the RF can be tuned by tuning these parameters.

> *Note*: I have yet to implement the `compute_trf` and `compute_erf` functions in the `UNet class` to compute the theoretical and effective receptive field, so I can start tuning and researching them.

## Preprocessing
Unzip the [datasets](#datasets-used) in the `preprocessed/raw` folder and run `python preprocess.py` with the following arguments.

| **Argument**                              |          **Flag**         | **Required** | **Type** |       **Default value**       |
|-------------------------------------------|:-------------------------:|:------------:|:--------:|:-----------------------------:|
| Dataset (`fetal_head` or `breast_cancer`) | `-d` or `--dataset`       |      yes     |    str   |                          None |
| Validation set rate                       | `-v` or `--val_rate`      |      no      |   float  |                           0.2 |
| Test set rate                             | `-t` or `--test_rate`     |      no      |   float  |                           0.1 |

## Training
Training the model can be done by running `python train.py -d <dataset>`. The dataset is the only required argument, all other arguments have default values as listed in this table:
| **Argument**                              |          **Flag**         | **Required** | **Type** |       **Default value**       |
|-------------------------------------------|:-------------------------:|:------------:|:--------:|:-----------------------------:|
| Dataset (`fetal_head` or `breast_cancer`) | `-d` or `--dataset`       |      yes     |    str   |                          None |
| Seed                                      | `-s` or `--seed`          |      no      |    int   |                            42 |
| Model config file                         | `-c` or `--config`        |      no      |    str   | "configurations/default.json" |
| Epochs                                    | `-e` or `--epochs`        |      no      |    int   |                            10 |
| Batch size                                | `-b` or `--batch_size`    |      no      |    int   |                             2 |
| Learning rate                             | `-l` or `--learning-rate` |      no      |   float  |                         0.001 |

After training, a checkpoint for each epoch will be saved at `checkpoints/{dataset_name}/{config_name}`. The last number of the file name corresponds to the validation loss at that epoch.

## Datasets Used
[**Fetal Head Ultrasound Images**](https://zenodo.org/record/1327317)<br>
Thomas L. A. van den Heuvel, Dagmar de Bruijn, Chris L. de Korte, & Bram van Ginneken. (2018). Automated measurement of fetal head circumference using 2D ultrasound images [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1327317

[**Breast Ultrasound Images Dataset (Dataset BUSI)**](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)<br>
Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.
