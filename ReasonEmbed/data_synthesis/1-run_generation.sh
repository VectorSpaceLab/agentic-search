#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

task_types=("biology" "earth_science" "economics" "psychology" "robotics" "stackoverflow" "sustainable_living" "leetcode" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" )

language="en"
num_samples=10000

cache_dir="${CACHE_DIR:-${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}}"
save_dir="${SYNTH_DATA_DIR:-./outputs/generated/Qwen2-5-72B-Instruct}"
generation_model="${GENERATION_MODEL:-Qwen2-5-72B-Instruct}"
generation_model_type="${GENERATION_MODEL_TYPE:-open-source}"
generation_port="${GENERATION_PORT:-8000}"

mkdir -p "$save_dir"

for task_type in "${task_types[@]}"; do
    echo "Generating for task_type: $task_type"

    cmd="python \"$script_dir/1-run_generation.py\" \
    --task_type $task_type \
    --save_dir $save_dir \
    --cache_dir $cache_dir \
    --language $language \
    --num_samples $num_samples \
    --model $generation_model \
    --model_type $generation_model_type \
    --port $generation_port \
    --num_processes 64 \
    --use_cleaned_corpus "

    echo "$cmd"
    eval "$cmd" 2>&1 | tee "$save_dir/log_${language}_${task_type}.txt"
done
