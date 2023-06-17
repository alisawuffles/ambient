#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --time=3:00:00
#SBATCH --output="slurm/filter/slurm-%J-%x.out"

cat $0
echo "--------------------"
data_file=generated_data/wanli_disagreement_p0.9_davinci-002/examples.jsonl

python -m filtering.filter \
    --seed_data_file data/wanli/train.jsonl \
    --data_file $data_file
