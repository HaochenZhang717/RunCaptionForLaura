#!/usr/bin/env bash
#SBATCH --job-name=graddiff-text
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
export CUDA_VISIBLE_DEVICES=2,3,4,5

SCRIPT=llava_gd.py

accelerate launch \
  --num_processes 4 \
  $SCRIPT \
  --model_id llava-hf/llava-1.5-7b-hf \
  --vanilla_dir llava-hf/llava-1.5-7b-hf \
  --save_dir ./ckpt/graddiff_text \
  --data_mode text_only \
  --forget_json VLGuard/train_forget_text_only.json \
  --retain_json VLGuard/train_retain_text_only.json \
  --batch_size 1

