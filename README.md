# GNNVerifier

Graph-based verifier to validate and correct issues in LLM-generated task planning.

## Get Started

First, install all the required packages:

```shell
conda create -n gnnverifier python=3.8
conda activate gnnverifier
pip install -r requirements.txt
```

Before running, set your **API** information in `main.sh`:

```shell
export OPENAI_API_KEY="your_openai_api_key_here"
export OPENAI_API_BASE="your_openai_api_base_here"
```

Then run the full pipeline (direct prediction, validation, and GNN verification with LLM refinement):

```shell
bash main.sh
```

The pipeline runs in order: `direct.py` → `direct_val.py` → `main.py`.

## Configurable Parameters

Edit the variables at the top of `main.sh` to customize runs:

| Variable      | Description                    | Example      |
|---------------|--------------------------------|--------------|
| `LLM_NAME`    | LLM for prediction & correction| `gpt-4o`     |
| `DATASET`     | Dataset name                   | `huggingface`|
| `LAMBDA_GRAPH`| Graph loss weight              | `2.0`        |
| `LR`          | Learning rate                  | `2e-4`       |
| `LAMBDA_GAP`  | Gap loss weight                | `1.5`        |
| `COST_TAU`    | Soft target temperature        | `0.6`        |

Supported datasets: `huggingface`, `multimedia`, `dailylife`, `ultratool`.
