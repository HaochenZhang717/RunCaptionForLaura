#!/usr/bin/env bash
#SBATCH --job-name=graddiff-mm
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
export CUDA_VISIBLE_DEVICES=2,3,4,5

SCRIPT=llava_gd.py

accelerate launch \
  --num_processes 4 \
  $SCRIPT \
  --model_id /common/$USER/models/llava-1.5-7b-hf \
  --vanilla_dir /common/$USER/models/llava-1.5-7b-hf \
  --save_dir ./ckpt/graddiff_mm \
  --forget_json VLGuard/train_forget.json \
  --retain_json VLGuard/train_retain.json \
  --image_root VLGuard/train_images/train \
  --batch_size 1