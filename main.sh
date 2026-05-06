#!/bin/bash

# ==============================
# Set your OpenAI API information
# ==============================
export HF_ENDPOINT=https://hf-mirror.com
# env | grep -i proxy
# unset ALL_PROXY all_proxy

# ==============================
# Configurable parameters
# ==============================

LLM_NAME="deepseek-chat"
DATASET="huggingface"

# Training hyperparameters (modifiable)
LAMBDA_GRAPH=2.0
LR=2e-4
LAMBDA_GAP=1.5
COST_TAU=0.6

# ==============================
# Run scripts
# ==============================

echo "========== Running direct.py =========="
python3 direct.py \
    --llm ${LLM_NAME} \
    --dataset ${DATASET} \
    --use_demos 1

echo "========== Running direct_val.py =========="
python3 direct_val.py \
    --llm ${LLM_NAME} \
    --dataset ${DATASET} \
    --use_demos 1

echo "========== Running main.py =========="
python3 main.py \
    --dataset ${DATASET} \
    --lambda_graph ${LAMBDA_GRAPH} \
    --lr ${LR} \
    --lambda_gap ${LAMBDA_GAP} \
    --cost_tau ${COST_TAU} \
    --llm_name ${LLM_NAME}
echo "========== All tasks finished =========="