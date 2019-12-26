#!/bin/bash
#SBATCH -J dnn_phone
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -x gqxx-01-025
#SBATCH -o exp/dnn_phone.%j
#SBATCH --gres=gpu:1
#SBATCH --mem=150G

dir=$1
fix=$2
encoder_path=$3
source /mnt/lustre/sjtu/home/zkz01/.extra
python runners/train_dnn.py --config config/config_dnn_mfcc.yaml --outputdir $dir --fix_encoder $fix
