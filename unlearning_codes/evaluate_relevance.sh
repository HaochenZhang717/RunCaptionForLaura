#!/bin/bash

export HF_HOME=/playpen/haochenz/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

export CUDA_VISIBLE_DEVICES=6

JUDGE_MODEL=Qwen/Qwen3-VL-8B-Instruct

########################################
# MULTIMODAL
########################################
python eval_relevance_qwen3vl.py \
  --input_json tuned_multimodal.json \
  --output_json tuned_multimodal_eval.json \
  --model_id $JUDGE_MODEL \
  --prompt_variant multimodal \
  --device cuda \
  --dtype auto \
  --max_new_tokens 256


########################################
# TEXT ONLY
########################################
python eval_relevance_qwen3vl.py \
  --input_json tuned_text_only.json \
  --output_json tuned_text_only_eval.json \
  --model_id $JUDGE_MODEL \
  --prompt_variant text_only \
  --device cuda \
  --dtype auto \
  --max_new_tokens 256


########################################
# IMAGE ONLY
########################################
python eval_relevance_qwen3vl.py \
  --input_json tuned_image_only.json \
  --output_json tuned_image_only_eval.json \
  --model_id $JUDGE_MODEL \
  --prompt_variant image_only \
  --device cuda \
  --dtype auto \
  --max_new_tokens 256