# Redapter Training Reproduction

This directory contains the Redapter training code used by ReasonEmbed. The code trains a decoder-only embedding model with dynamic QRI-weighted contrastive learning.

The reference environment used to create `requirements.txt` was:

- Python 3.10.16
- CUDA 12.4
- PyTorch 2.6.0
- FlagEmbedding at commit `dbc600560b2dadcc1514989092f7b849673bb67d`

## 1. Clone The Repository

```bash
git clone https://github.com/VectorSpaceLab/agentic-search.git
cd agentic-search/ReasonEmbed/training
```

## 2. Install The Environment

Create a Python 3.10 environment, then install the pinned dependencies.

```bash
conda create -n reasonembed-train python=3.10 -y
conda activate reasonembed-train
pip install -r requirements.txt
```

`requirements.txt` installs `FlagEmbedding` directly from `https://github.com/FlagOpen/FlagEmbedding.git` at commit `dbc600560b2dadcc1514989092f7b849673bb67d`, because the training code depends on APIs from that revision.

If `flash-attn` fails to build on your machine, install the wheel that matches your CUDA and PyTorch versions first, then rerun the remaining installation.

## 3. Download The Training Data

The released training data is hosted at `hanhainebula/reason-embed-data`. Download the paper training data and point `--train_data` to the local directory.

```bash
mkdir -p ./data
huggingface-cli download hanhainebula/reason-embed-data \
  --repo-type dataset \
  --include "reason-embed-data-0928/*" \
  --local-dir ./data/reason-embed-data
```

The training examples should contain the fields consumed by `custom_dataset.py`, including `prompt`, `query`, `reasoning_query`, `pos`, and `neg`. Optional fields such as `train_group_size`, `batch_size`, `pos_scores`, and `neg_scores` are handled by the FlagEmbedding data pipeline when present.

## 4. Configure The Training Inputs

`train.sh` reads the following four values from environment variables. Set them before launching training:

```bash
export MODEL_NAME_OR_PATH=hanhainebula/qwen3-8b-ft-msmarco
export TRAIN_DATA=./data/reason-embed-data/reason-embed-data-0928
export CACHE_DIR=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export CACHE_PATH=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
```

You can replace `MODEL_NAME_OR_PATH` with a local checkpoint path if the base model has already been downloaded.

## 5. Modify The Training Script

Use `train.sh` as the reference script. The four path/model values above are configured with environment variables; the remaining training hyperparameters are fixed in the script. Before running on a new machine, review these script-local settings if your hardware requires changes:

- `output_dir`: choose where checkpoints and logs should be written.
- `num_gpus`: set this to the number of GPUs available on the node.
- `per_device_train_batch_size`: reduce this if GPU memory is insufficient.
- `--deepspeed`: defaults to `./ds_stage1.json` in this directory.

The main hyperparameters in the reference script are:

- LoRA rank `32` and LoRA alpha `64`
- `train_group_size 16`
- `query_max_len 512` and `passage_max_len 512`
- `temperature 0.02`
- `sentence_pooling_method last_token`
- `normalize_embeddings True`
- `qri_start_step 100`
- `qri_score_mapping clamp`

## 6. Run Training

After updating `train.sh`, launch training from this directory:

```bash
# activate environment (example)
source /root/anaconda3/bin/activate reasonembed-train

# configure train.sh inputs
export MODEL_NAME_OR_PATH=hanhainebula/qwen3-8b-ft-msmarco
export TRAIN_DATA=./data/reason-embed-data/reason-embed-data-0928
export CACHE_DIR=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export CACHE_PATH=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}

# run training
bash train.sh
```

The script runs `torchrun --nproc_per_node $num_gpus main.py ...`. When `--save_merged_lora_model True` is enabled, the merged model is saved under the output directory after training.

For a quick smoke test, lower `num_gpus`, reduce `per_device_train_batch_size`, and point `train_data` to a small subset of the released data before running a full experiment.

## 7. Evaluate The Trained Model

The BRIGHT short evaluation follows the same pattern as the released `evaluation_scripts/eval_bright_short.sh` script. Replace `--embedder_name_or_path` with the path to your merged checkpoint:

```bash
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
python -m FlagEmbedding.evaluation.bright \
  --embedder_name_or_path /path/to/output_dir/merged_model \
  --embedder_model_class decoder-only-base \
  --query_instruction_format_for_retrieval 'Instruct: {}\nQuery: {}' \
  --pooling_method last_token \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  --cache_dir $HF_HUB_CACHE \
  --embedder_batch_size 8 \
  --embedder_query_max_length 8192 \
  --embedder_passage_max_length 8192 \
  --task_type short \
  --use_special_instructions True \
  --eval_name bright_short \
  --dataset_dir ./bright_short/data \
  --dataset_names biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_questions theoremqa_theorems \
  --splits examples \
  --corpus_embd_save_dir ./bright_short/corpus_embd \
  --output_dir ./bright_short/search_results/examples \
  --search_top_k 2000 \
  --cache_path $HF_HUB_CACHE \
  --overwrite False \
  --k_values 1 10 100 \
  --eval_output_method markdown \
  --eval_output_path ./bright_short/eval_results_examples.md \
  --eval_metrics ndcg_at_10 recall_at_10 recall_at_100
```

The important evaluation settings are:

- `--embedder_model_class decoder-only-base`
- `--query_instruction_format_for_retrieval 'Instruct: {}\nQuery: {}'`
- `--pooling_method last_token`
- `--embedder_query_max_length 8192`
- `--embedder_passage_max_length 8192`
- `--task_type short`
- `--eval_name bright_short`
- `--eval_metrics ndcg_at_10 recall_at_10 recall_at_100`

Make sure `--dataset_dir`, `--corpus_embd_save_dir`, and `--output_dir` point to writable local directories. The reference script uses eight GPUs via `--devices cuda:0 ... cuda:7`; change this list to match your hardware.
