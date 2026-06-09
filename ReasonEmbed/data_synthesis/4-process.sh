#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

data_dir="${ANNOTATION_OUTPUT_DIR:-./outputs/labeled}"
save_dir="${PROCESSED_DATA_DIR:-./outputs/processed}"
output_name="${OUTPUT_NAME:-annotator-qwen3-8b-0928}"
label_llm="${LABEL_LLM:-reason-embed-annotator-qwen3-8b-0928}"

task_types=("biology" "earth_science" "economics" "psychology" "robotics" "stackoverflow" "sustainable_living" "leetcode" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" )

for i in "${!task_types[@]}"; do
    task_type=${task_types[$i]}
    
    echo "Processing task_type: $task_type"

    cmd="python \"$script_dir/4-process.py\" \
    --task_type $task_type \
    --data_file $data_dir/$output_name/$task_type-$label_llm-labeled.jsonl \
    --output_file $save_dir/$output_name/$task_type-formatted.jsonl \
    "

    echo "$cmd"
    eval "$cmd"
    echo "Finished processing task_type: $task_type"
done
