# ReMixer Data Synthesis Reproduction

This directory contains the ReMixer data synthesis pipeline used by ReasonEmbed. The scripts generate reasoning-intensive synthetic queries from BRIGHT corpora, mine hard negatives, label candidate documents with the ReasonEmbed annotator, and format the final training data.

The reference environment used to create `requirements.txt` was:

- Python 3.10.16
- CUDA 12.4
- PyTorch 2.6.0
- FlagEmbedding at commit `dbc600560b2dadcc1514989092f7b849673bb67d`
- SGLang 0.4.6.post4

## 1. Clone The Repository

```bash
git clone https://github.com/VectorSpaceLab/agentic-search.git
cd agentic-search/ReasonEmbed/data_synthesis
```

## 2. Install The Environment

Create a Python 3.10 environment, then install the pinned dependencies.

```bash
conda create -n reasonembed-synthesis python=3.10 -y
conda activate reasonembed-synthesis
pip install -r requirements.txt
```

`requirements.txt` installs `FlagEmbedding` directly from `https://github.com/FlagOpen/FlagEmbedding.git` at commit `dbc600560b2dadcc1514989092f7b849673bb67d`, because the data synthesis and training code depend on APIs from that revision.

The provided lock file uses `faiss-gpu==1.7.3`, matching the reference environment. If you run only CPU FAISS, replace it with a compatible `faiss-cpu` package for your platform.

## 3. Prepare BRIGHT Data

Download or prepare the BRIGHT short benchmark data so that each domain has files such as `corpus.jsonl` and `examples_qrels.jsonl`.

```bash
export BRIGHT_DATA_ROOT=./bright_short/data
```

The expected domain subdirectories are:

```text
biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_questions theoremqa_theorems
```

## 4. Configure Paths And Models

The shell scripts read paths and model names from environment variables. Defaults are relative to this directory, but setting them explicitly makes the run easier to reproduce:

```bash
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export CACHE_DIR=$HF_DATASETS_CACHE
export MODEL_CACHE_DIR=$HF_HUB_CACHE

export BRIGHT_DATA_ROOT=./bright_short/data
export SYNTH_DATA_DIR=./outputs/generated/Qwen2-5-72B-Instruct
export SYNTH_MINED_DIR=./outputs/mined/Qwen2-5-72B-Instruct
export SYNTH_INDEX_DIR=./outputs/indexes
export ANNOTATION_INPUT_DIR=./outputs/to-be-labeled
export ANNOTATION_OUTPUT_DIR=./outputs/labeled
export PROCESSED_DATA_DIR=./outputs/processed

export GENERATION_MODEL=Qwen2-5-72B-Instruct
export GENERATION_MODEL_TYPE=open-source
export GENERATION_PORT=8000
export VLLM_HOST=localhost

export EMBEDDER_NAME_OR_PATH=Alibaba-NLP/gte-Qwen2-7B-instruct
export ANNOTATOR_MODEL_NAME_OR_PATH=hanhainebula/reason-embed-annotator-qwen3-8b-0928
export OUTPUT_NAME=annotator-qwen3-8b-0928
export LABEL_LLM=reason-embed-annotator-qwen3-8b-0928
```

For `GENERATION_MODEL_TYPE=open-source`, start a vLLM-compatible OpenAI API server before running generation. You can also set `VLLM_BASE_URL` directly if the endpoint is not `http://$VLLM_HOST:$GENERATION_PORT/v1/`.

## 5. Clean Corpus For Generation

This optional step filters BRIGHT corpus documents by asking the generation model whether each document belongs to the target domain. It writes `annotated_for_generation/corpus_annotated_true.jsonl` under each BRIGHT domain directory.

```bash
python 0-clean_corpus_for_generation.py
```

`1-run_generation.sh` uses the cleaned corpus by default through `--use_cleaned_corpus`. Skip this step only if you also remove that flag from the script.

## 6. Generate Synthetic Triplets

Run query and reasoning-query generation for all BRIGHT domains:

```bash
bash 1-run_generation.sh
```

The script calls `1-run_generation.py` in this directory and writes outputs under `$SYNTH_DATA_DIR`.

## 7. Mine Hard Negatives

Use `Alibaba-NLP/gte-Qwen2-7B-instruct` by default to mine hard negatives from the BRIGHT corpus:

```bash
bash 2-mine_docs_for_bright.sh
```

The script calls `2-mine_docs_for_bright.py`, reads generated triplets from `$SYNTH_DATA_DIR`, reads candidate documents from `$BRIGHT_DATA_ROOT`, and writes mined triplets to `$SYNTH_MINED_DIR`.

## 8. Label Candidate Documents

Point `$ANNOTATION_INPUT_DIR` to the files that should be labeled, then run:

```bash
bash 3-label.sh
```

The script calls `3-label.py` and writes labeled files to `$ANNOTATION_OUTPUT_DIR/$OUTPUT_NAME`. By default it uses `hanhainebula/reason-embed-annotator-qwen3-8b-0928`; replace `$ANNOTATOR_MODEL_NAME_OR_PATH` with a local model path if needed.

## 9. Format Training Data

Convert labeled files into the final ReasonEmbed training format:

```bash
bash 4-process.sh
```

The script calls `4-process.py` and writes formatted files to `$PROCESSED_DATA_DIR/$OUTPUT_NAME`. The output examples contain fields such as `prompt`, `query`, `pos`, `neg`, `train_group_size`, and `batch_size`, which are consumed by the Redapter training code.
