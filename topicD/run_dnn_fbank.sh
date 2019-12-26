#!/bin/bash
#SBATCH -J dnn_phone
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -x gqxx-01-025
#SBATCH -o exp/dnn_phone.%j
#SBATCH --gres=gpu:1
#SBATCH --mem=150G

source /mnt/lustre/sjtu/home/zkz01/.extra
fix=true
python runners/train_dnn.py --config config/config_dnn_fbank.yaml --outputdir exp/dnn_212_fbank_$fix --fix_encoder $fix --encoder_path exp/ae_212_fbank/model.pth
