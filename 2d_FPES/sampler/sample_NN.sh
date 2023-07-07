#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbi_env

python ${1:-"/home/mrtodd/2d_FPES/sample_nn/sample_NN.py"} ${2:-100} 
