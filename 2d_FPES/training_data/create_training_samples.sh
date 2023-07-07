#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbi_env

rm /home/mrtodd/2d_FPES/training_data/injection.hdf
rm /home/mrtodd/2d_FPES/training_data/training_samples.hdf

bash /home/mrtodd/2d_FPES/training_data/create_injection.sh ${1:-"home/mrtodd/2d_FPES/training_data/injection.ini"} ${2:-1000}

python /home/mrtodd/2d_FPES/training_data/create_training_samples.py "/home/mrtodd/2d_FPES/training_data/injection.hdf"


