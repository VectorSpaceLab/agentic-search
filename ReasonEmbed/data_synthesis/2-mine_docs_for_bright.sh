#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

data_dir="${SYNTH_DATA_DIR:-./outputs/generated/Qwen2-5-72B-Instruct}"
corpus_dir="${BRIGHT_DATA_ROOT:-./bright_short/data}"
save_dir="${SYNTH_MINED_DIR:-./outputs/mined/Qwen2-5-72B-Instruct}"
index_save_dir="${SYNTH_INDEX_DIR:-./outputs/indexes}"
model_cache_dir="${MODEL_CACHE_DIR:-${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}}"
embedder_name_or_path="${EMBEDDER_NAME_OR_PATH:-Alibaba-NLP/gte-Qwen2-7B-instruct}"

task_types=("biology" "earth_science" "economics" "psychology" "robotics" "stackoverflow" "sustainable_living" "leetcode" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" )

language="en"

for task_type in "${task_types[@]}"; do
    echo "Processing task_type: $task_type"

    cmd="python \"$script_dir/2-mine_docs_for_bright.py\" \
    --embedder_name_or_path $embedder_name_or_path \
    --embedder_model_class decoder-only-base \
    --trust_remote_code True \
    --pooling_method last_token \
    --cache_dir $model_cache_dir \
    --query_instruction_format_for_retrieval 'Instruct: {}\nQuery: {}' \
    --batch_size 8 \
    --embedder_query_max_length 8192 \
    --embedder_passage_max_length 8192 \
    --domain $task_type \
    --input_file $data_dir/$language/$task_type/$language-triplets.jsonl \
    --output_file $save_dir/$task_type-$language-triplets.jsonl \
    --candidate_pool $corpus_dir/$task_type/corpus.jsonl \
    --index_save_dir $index_save_dir/gte-Qwen2-7B-instruct/$task_type \
    --search_top_k 1000 \
    --candidates_number 100 \
    --use_gpu_for_searching True "

    echo "$cmd"
    eval "$cmd"
    echo "Finished processing task_type: $task_type"
done
