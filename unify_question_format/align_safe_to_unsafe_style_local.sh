#!/usr/bin/env bash


SCRIPT_DIR="/Users/zhc/Documents/PhD/projects/RunCaptionForLaura/unify_question_format"
cd "$SCRIPT_DIR"

export HF_HOME=/playpen/haochenz/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"


python align_safe_to_unsafe_style.py \
  --model "$MODEL" \
  --files test.json \
  --out_suffix "_safe_style_matched" \
  --batch_size 16 \
  --max_new_tokens 96 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 0
