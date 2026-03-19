#!/usr/bin/env bash
#SBATCH --job-name=graddiff-img
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
export CUDA_VISIBLE_DEVICES=2,3,4,5
SCRIPT=llava_gd.py

accelerate launch \
  --num_processes 4 \
  $SCRIPT \
  --model_id /common/$USER/models/llava-1.5-7b-hf \
  --vanilla_dir /common/$USER/models/llava-1.5-7b-hf \
  --save_dir ./ckpt/graddiff_img \
  --forget_json VLGuard/train_forget_image_only_3_sentence.json \
  --retain_json VLGuard/train_retain_image_only_3_sentence.json \
  --image_root VLGuard/train_images/train \
  --batch_size 1 \
  --gamma 1.0 \
  --retain_loss_weight 1.0
