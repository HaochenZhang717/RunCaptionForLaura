#!/usr/bin/env bash
#SBATCH --job-name=graddiff-text
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
export CUDA_VISIBLE_DEVICES=2,3,4,5
export HF_HOME=/playpen/haochenz/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

SCRIPT=llava_gd.py

#accelerate launch \
#  --num_processes 4 \
#  $SCRIPT \
#  --model_id llava-hf/llava-1.5-7b-hf \
#  --vanilla_dir llava-hf/llava-1.5-7b-hf \
#  --save_dir ./ckpt/graddiff_text \
#  --data_mode text_only \
#  --forget_json "/playpen-shared/laura/unlearning/VLGuard/train_forget_image_only_3_sentence.json" \
#  --retain_json "/playpen-shared/laura/unlearning/VLGuard/train_retain_image_only_3_sentence.json" \
#  --image_root "/playpen-shared/laura/unlearning/VLGuard/train_images/train" \
#  --batch_size 1 \
#  --gamma 1.0 \
#  --retain_loss_weight 1.0

accelerate launch \
  --num_processes 4 \
  $SCRIPT \
  --model_id llava-hf/llava-1.5-7b-hf \
  --vanilla_dir llava-hf/llava-1.5-7b-hf \
  --save_dir ./ckpt/graddiff_text \
  --data_mode text_only \
  --forget_json "/playpen-shared/laura/unlearning/VLGuard/train_forget_text_only.json" \
  --retain_json "/playpen-shared/laura/unlearning/VLGuard/train_retain_text_only.json" \
  --image_root "/playpen-shared/laura/unlearning/VLGuard/train_images/train" \
  --batch_size 1 \
  --gamma 1.0 \
  --retain_loss_weight 1.0
