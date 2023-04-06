# Study of Receptive Field of U-Net and Attention U-Net 

The U-Net in this code consists of 4 encoder blocks, a bottleneck, and 4 decoder blocks. All parameters for all layers in all blocks can be easily tuned by creating a configuration file and passing it into the UNet class. I have already created a [default configuration file](configurations/default.py) to test the UNet. This is very useful as the receptive field depends on the kernel size, stride and padding of each layer; so the RF can be tuned by tuning these parameters.

> *Note*: I have yet to implement the `compute_trf` and `compute_erf` functions in the `UNet class` to compute the theoretical and effective receptive field, so I can start tuning and researching them.

## Datasets Used
[**Fetal Head Ultrasound Images**](https://zenodo.org/record/1327317)<br>
Thomas L. A. van den Heuvel, Dagmar de Bruijn, Chris L. de Korte, & Bram van Ginneken. (2018). Automated measurement of fetal head circumference using 2D ultrasound images [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1327317

[**Breast Ultrasound Images Dataset (Dataset BUSI)**](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)<br>
Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.
