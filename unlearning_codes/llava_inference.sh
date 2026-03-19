#!/bin/bash
export HF_HOME=/playpen/haochenz/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export CUDA_VISIBLE_DEVICES=6


python llava_inference.py \
  --base_model llava-hf/llava-1.5-7b-hf \
  --model_path ./ckpt/graddiff_mm \
  --input_json /playpen-shared/laura/unlearning/safety/vlguard_test_qwen_input.json \
  --image_root /playpen-shared/laura/unlearning/VLGuard/test_images/test \
  --mode multimodal \
  --output_name tuned_multimodal.json


python llava_inference.py \
  --base_model llava-hf/llava-1.5-7b-hf \
  --model_path ./ckpt/graddiff_text \
  --input_json /playpen-shared/laura/unlearning/safety/vlguard_test_qwen_input_text_only.json \
  --mode text_only \
  --output_name tuned_text_only.json



python llava_inference.py \
  --base_model llava-hf/llava-1.5-7b-hf \
  --model_path ./ckpt/graddiff_img \
  --input_json /playpen-shared/laura/unlearning/safety/vlguard_test_qwen_captioning.json \
  --image_root /playpen-shared/laura/unlearning/VLGuard/test_images/test \
  --mode image_only \
  --output_name tuned_image_only.json