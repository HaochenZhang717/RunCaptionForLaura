#!/bin/bash

############################################
# GPU CONFIG
############################################

export CUDA_VISIBLE_DEVICES=0,1,2,4
NUM_GPU=1

############################################
# DISTRIBUTED SETTINGS
############################################

MASTER_ADDR=127.0.0.1
MASTER_PORT=29549

############################################
# NCCL SETTINGS (important for stability)
############################################

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

############################################
# TRANSFORMERS CACHE (optional but recommended)
############################################

export HF_HOME=/playpen-nvme/haochenz/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

############################################
# PYTHON
############################################

PYTHON_SCRIPT=grad_diff_qwen.py

############################################
# OUTPUT
############################################

LOG_DIR=logs
mkdir -p $LOG_DIR

LOG_FILE=$LOG_DIR/graddiff_qwen3vl_$(date +%Y%m%d_%H%M%S).log

############################################
# RUN
############################################

torchrun \
--nnodes=1 \
--nproc_per_node=$NUM_GPU \
--master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT \
$PYTHON_SCRIPT \
2>&1 | tee $LOG_FILE



scp -r haochenz@unites6.cs.unc.edu:/playpen-shared/laura/unlearning/VLGuard/test_images/test/bad_ads/ed926a06-4d80-4e3a-9c22-225c232f3d5c.png /Users/zhc/Downloads/
scp -r haochenz@unites6.cs.unc.edu:/playpen-shared/laura/unlearning/VLGuard/test_images/test/bad_ads/0245d9d5-eed7-4a4f-935a-fb587071da13.png /Users/zhc/Downloads/
scp -r haochenz@unites6.cs.unc.edu:/playpen-shared/laura/unlearning/VLGuard/test_images/test/bad_ads/073b8ea5-cc7c-4997-8f42-a75736cac276.png /Users/zhc/Downloads/