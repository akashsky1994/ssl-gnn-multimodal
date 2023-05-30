#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=24:10:00
#SBATCH --mem=64GB
#SBATCH --job-name=vgae_classifier
#SBATCH --output=logs/vgae_classifier_%j.out

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate ssl-gnn-env

cd /scratch/am11533/ssl-gnn-multimodal

python ssl_gnn_multimodal/main.py --model GMAE --dataset HATEFULMEME --data_path ../datasets/hateful_memes/  --resume ./checkpoints/GMAE/pretrain/

# ``nohup python -u ssl_gnn_multimodal/main.py --model GMAE --data_path ../datasets/hateful_memes/  --resume ./checkpoints/GMAE/pretrain/ > output_gmae_pretrain_$(date +%s).log &``