#!/bin/bash
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=80:00:00
#SBATCH --output="slurm/generate/slurm-%J.out"

cat $0
echo "--------------------"

python -m generation.generation_pipeline \
    --model_path models/roberta-large-wanli \
    --num_gens_per_prompt 5 \
    --num_incontext_examples 4 \
    --top_p 0.9 \
    --engine 'text-davinci-002' \
    --num_examples -1
