#!/bin/bash
#SBATCH --job-name=continuation_eval
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:2
#SBATCH --time=30:00:00
#SBATCH --output="slurm/evaluate/slurm-%J-%x.out"

cat $0
echo "--------------------"
echo "shard ${shard}"

python -m evaluation.continuation_evaluation \
    --data_path annotation/AmbiEnt/llama_missing_shards/test_${shard}.jsonl \
    --model_name llama-65b \
    --top_p 1.0 \
    --num_generations 100
