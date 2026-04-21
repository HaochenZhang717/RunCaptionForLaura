#!/usr/bin/env bash
#SBATCH --job-name=safe-style-match
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/Users/zhc/Documents/PhD/projects/RunCaptionForLaura/unify_question_format/logs/%x-%j.out
#SBATCH --error=/Users/zhc/Documents/PhD/projects/RunCaptionForLaura/unify_question_format/logs/%x-%j.err

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-YOUR_CONDA_ENV}"
if [[ "$CONDA_ENV" == "YOUR_CONDA_ENV" ]]; then
  echo "Please set CONDA_ENV before submitting, for example:" >&2
  echo "  sbatch --export=ALL,CONDA_ENV=myenv align_safe_to_unsafe_style_slurm.sh" >&2
  exit 1
fi

source ~/.bashrc
conda activate "$CONDA_ENV"

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
