#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=snellius-notifications.tnhs2@8shield.net

module load 2021
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
module load torchvision/0.11.1-foss-2021a-CUDA-11.3.1

pip install dotmap
pip install tqdm
pip install opencv_python
pip install matplotlib

# Copy input file to scratch
cp -r $HOME/bsc-thesis/data $TMPDIR
mkdir $TMPDIR/output_dir
 
# Execute program located in $HOME
python $HOME/train.py -o $TMPDIR/outout_dir -d breast_cancer -c trf50 -b 2 -e 2 -l 0.01

# Copy output directory from scratch to home
cp -r $TMPDIR/output_dir $HOME/bsc-thesis