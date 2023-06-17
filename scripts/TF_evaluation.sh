#!/bin/bash
#SBATCH --job-name=TF_eval
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --output="slurm/evaluate/slurm-%J-%x.out"

cat $0
echo "--------------------"

python -m evaluation.TF_evaluation \
    --model_name gpt-4 \
    --data_path annotation/AmbiEnt/test.jsonl \
