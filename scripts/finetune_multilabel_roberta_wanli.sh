#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --output="slurm/train/slurm-%J-%x.out"

cat $0
echo "--------------------"

train_file=data/wanli/multilabel_train.jsonl
output_dir=models/roberta-large-wanli-multilabel

python -m classification.run_multilabel_nli \
  --model_name_or_path roberta-large \
  --do_train \
  --train_file $train_file \
  --per_device_train_batch_size 32 \
  --save_strategy epoch \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --overwrite_cache \
  --output_dir $output_dir
