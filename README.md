# Study of Receptive Field of U-Net and Attention U-Net 

## Datasets
**TODO**


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
