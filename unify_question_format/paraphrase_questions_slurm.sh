#!/usr/bin/env bash
#SBATCH --job-name=paraphrase-q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

export HF_HOME=/playpen/haochenz/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

MODEL="Qwen/Qwen2.5-7B"

python paraphrase_questions.py \
  --model "$MODEL" \
  --files test.json train_forget.json train_retain.json \
  --target_form "instruction" \
  --batch_size 16 \
  --max_new_tokens 128 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 0

python paraphrase_questions.py \
  --model "$MODEL" \
  --files test.json train_forget.json train_retain.json \
  --target_form "question" \
  --batch_size 16 \
  --max_new_tokens 128 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 0
