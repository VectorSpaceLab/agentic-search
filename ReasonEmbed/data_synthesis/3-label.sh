#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

data_dir="${ANNOTATION_INPUT_DIR:-./outputs/to-be-labeled}"
save_dir="${ANNOTATION_OUTPUT_DIR:-./outputs/labeled}"
cache_dir="${CACHE_DIR:-${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}}"
annotator_model_name_or_path="${ANNOTATOR_MODEL_NAME_OR_PATH:-hanhainebula/reason-embed-annotator-qwen3-8b-0928}"
output_name="${OUTPUT_NAME:-annotator-qwen3-8b-0928}"

task_types=("biology" "earth_science" "economics" "psychology" "robotics" "stackoverflow" "sustainable_living" "leetcode" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" )

for i in "${!task_types[@]}"; do
    task_type=${task_types[$i]}
    
    echo "Processing task_type: $task_type"

    cmd="python \"$script_dir/3-label.py\" \
    --model_name_or_path $annotator_model_name_or_path \
    --context_length 40960 \
    --tp_size 1 \
    --dp_size 8 \
    --batch_size 80 \
    --data_file $data_dir/$task_type-to-be-labeled.jsonl \
    --cache_dir $cache_dir \
    --output_name $output_name \
    --task_type $task_type \
    --output_dir $save_dir \
    --top_k 100 \
    "

    echo "$cmd"
    eval "$cmd"
    echo "Finished processing task_type: $task_type"
done
