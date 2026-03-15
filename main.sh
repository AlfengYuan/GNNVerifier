#!/bin/bash

# ==============================
# Set your OpenAI API information
# ==============================
export OPENAI_API_KEY="your_openai_api_key_here"
export OPENAI_API_BASE="your_openai_api_base_here"

# ==============================
# Configurable parameters
# ==============================

LLM_NAME="gpt-4o"
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