#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbi_env

python ${1:-"/home/mrtodd/1d_FPES/plots/make_plots.py"} $2 $3

