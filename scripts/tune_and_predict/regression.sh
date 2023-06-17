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
echo "$model"

python -m classification.tune_and_predict.regression_nli \
    --model_path $model
