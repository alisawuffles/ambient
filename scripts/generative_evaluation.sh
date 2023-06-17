#!/bin/bash
#SBATCH --job-name=generative_eval
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --output="slurm/evaluate/slurm-%J-%x.out"

cat $0
echo "--------------------"

python -m evaluation.generative_evaluation \
    --data_path annotation/AmbiEnt/test.jsonl \
    --model_name flan-t5-xxl \
    --num_incontext_examples 4
