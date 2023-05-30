#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --time=24:10:00
#SBATCH --mem=64GB
#SBATCH --job-name=gmae_cc
#SBATCH --output=logs/gmae_gat_cc_%j.out

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate ssl-gnn-env

cd /scratch/am11533/ssl-gnn-multimodal

python ssl_gnn_multimodal/main.py --pretrain --model GMAE --dataset CONCEPTUALCAPTION --data_path ../datasets/cc12m/  --resume ./checkpoints/GMAE/pretrain/ 