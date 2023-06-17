#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --output="slurm/predict/slurm-%J-%x.out"

cat $0
echo "--------------------"

python -m classification.tune_and_predict.binary_nli \
    --ent_model_path $ent_model \
    --neu_model_path $neu_model \
    --con_model_path $con_model \
