# Study of Receptive Field of U-Net and Attention U-Net 

## Preprocessing
Unzip the [datasets](#datasets-used) in the `preprocessed/raw` folder and run `python preprocess.py` with the following arguments.

| **Argument**                              |          **Flag**         | **Required** | **Type** |       **Default value**       |
|-------------------------------------------|:-------------------------:|:------------:|:--------:|:-----------------------------:|
| Dataset (`fetal_head` or `breast_cancer`) | `-d` or `--dataset`       |      yes     |    str   |                          None |
| Validation set rate                       | `-v` or `--val_rate`      |      no      |   float  |                          0.15 |
| Test set rate                             | `-t` or `--test_rate`     |      no      |   float  |                          0.15 |

## Training
Training the model can be done by running `python train.py -d <dataset>`. The dataset is the only required argument, all other arguments have default values as listed in this table:
| **Argument**                              |          **Flag**         | **Required** | **Type** |       **Default value**       |
|-------------------------------------------|:-------------------------:|:------------:|:--------:|:-----------------------------:|
| Dataset (`fetal_head` or `breast_cancer`) | `-d` or `--dataset`       |      yes     |    str   |                          None |
| Seed                                      | `-s` or `--seed`          |      no      |    int   |                            42 |
| Model config file                         | `-c` or `--config`        |      no      |    str   |                     "default" |
| Epochs                                    | `-e` or `--epochs`        |      no      |    int   |                           100 |
| Batch size                                | `-b` or `--batch_size`    |      no      |    int   |                             2 |
| Learning rate                             | `-l` or `--learning_rate` |      no      |   float  |                          0.01 |
| Output folder                             | `-o` or `--output_dir`    |      no      |    str   |                      "output" |

You can run `python generate_experiment.py` to generate a command which runs all possible experiments. By default it uses a batch size of 2 and a maximum of 100 epochs.

After training, the best model for each dataset and configuration is stored in `out/{dataset_name}/{config_name}`.

## Datasets Used
[**Fetal Head Ultrasound Images**](https://zenodo.org/record/1327317)<br>
Thomas L. A. van den Heuvel, Dagmar de Bruijn, Chris L. de Korte, & Bram van Ginneken. (2018). Automated measurement of fetal head circumference using 2D ultrasound images [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1327317

[**Breast Ultrasound Images Dataset (Dataset BUSI)**](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)<br>
Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.

[**Mouse Enbryo Ultrasound Dataset**](https://ieee-dataport.org/open-access/ultrasound-mouse-embryo-segmentation-and-classification)
ziming qiu, Tongda Xu, Jack Langerman, William Das, Chuiyu Wang, Nitin Nair, Orlando Aristizábal, Jonathan Mamou, Daniel H. Turnbull, Jeffrey A. Ketterling, Yao Wang. (2021). Ultrasound Mouse Embryo Segmentation and Classification. IEEE Dataport. https://dx.doi.org/10.21227/drmd-xq36 
