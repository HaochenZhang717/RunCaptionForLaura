#!/bin/bash

############################################
# GPU CONFIG
############################################

export CUDA_VISIBLE_DEVICES=1,2,3,4
NUM_GPU=4

############################################
# DISTRIBUTED SETTINGS
############################################

MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

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

export HF_HOME=/playpen/haochenz/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

############################################
# PYTHON
############################################

PYTHON_SCRIPT=train_qwen3vl_graddiff.py

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