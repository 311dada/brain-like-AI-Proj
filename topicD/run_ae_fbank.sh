#!/bin/bash
#SBATCH -J auto_encoder
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -x gqxx-01-025
#SBATCH -o exp/ae.%j
#SBATCH --gres=gpu:1
#SBATCH --mem=150G


dir=$1
source /mnt/lustre/sjtu/home/zkz01/.extra

export PYTHONPATH=/mnt/lustre/sjtu/home/zkz01/leinao:$PYTHONPATH
python runners/train_autoencoder.py --config config/config_autoencoder_fbank.yaml --outputdir $dir
