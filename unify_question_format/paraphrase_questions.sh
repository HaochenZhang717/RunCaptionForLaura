#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/playpen/haochenz/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME


MODEL="Qwen/Qwen2.5-7B-Instruct"

python paraphrase_questions.py \
  --model "$MODEL" \
  --files test.json train_forget.json train_retain.json \
  --target_form "instruction" \
  --batch_size 16 \
  --max_new_tokens 128 \
  --temperature 0.7 \
  --top_p 0.9 \
  --seed 0


python paraphrase_questions.py \
  --model "$MODEL" \
  --files test.json train_forget.json train_retain.json \
  --target_form "question" \
  --batch_size 16 \
  --max_new_tokens 128 \
  --temperature 0.7 \
  --top_p 0.9 \
  --seed 0
