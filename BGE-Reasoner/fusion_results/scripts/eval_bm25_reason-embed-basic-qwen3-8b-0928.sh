#!/bin/bash

# get code directory (two levels up)
code_root=$(dirname "$(dirname "$0")")
code_path="$code_root/code/fusion_results.py"

# get bright search results root directory from command line argument
bright_search_results_root="$1/reasoner-rewritten-query-0821"

bm25_results_path="$bright_search_results_root/bm25/NoReranker"

embedder_results_path="$bright_search_results_root/reason-embed-basic-qwen3-8b-0928/NoReranker"

save_dir="$bright_search_results_root/fusion/reproduction/bm25_reason-embed-basic-qwen3-8b-0928"


cmd="python $code_path \
    --results_dirs \
        $bm25_results_path \
        $embedder_results_path \
    --weights 0.25 0.75 \
    --merged_model_name bm25_reason-embed-basic-qwen3-8b-0928 \
    --top_k 2000 \
    --ori_split_name reasoner-rewritten-query-0821 \
    --save_dir $save_dir \
    --norm "

echo "Executing command: $cmd"
eval $cmd
