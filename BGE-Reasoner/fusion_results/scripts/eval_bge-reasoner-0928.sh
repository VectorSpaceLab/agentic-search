#!/bin/bash

# get code directory (two levels up)
code_root=$(dirname "$(dirname "$0")")
code_path="$code_root/code/fusion_results.py"

# get bright search results root directory from command line argument
bright_search_results_root="$1/reasoner-rewritten-query-0821"

bm25_embedder_results_path="$bright_search_results_root/fusion/bm25_reason-embed-basic-qwen3-8b-0928"

bm25_reranker_8b_results_path="$bright_search_results_root/bm25/retro-star-qwen3-8b-0928"
bm25_reranker_14b_results_path="$bright_search_results_root/bm25/retro-star-qwen3-14b-0928"
bm25_reranker_32b_results_path="$bright_search_results_root/bm25/retro-star-qwen3-32b-0928"

embedder_reranker_8b_results_path="$bright_search_results_root/reason-embed-basic-qwen3-8b-0928/retro-star-qwen3-8b-0928"
embedder_reranker_14b_results_path="$bright_search_results_root/reason-embed-basic-qwen3-14b-0928/retro-star-qwen3-14b-0928"
embedder_reranker_32b_results_path="$bright_search_results_root/reason-embed-basic-qwen3-32b-0928/retro-star-qwen3-32b-0928"

save_dir="$bright_search_results_root/fusion/reproduction/bge-reasoner-0928"


cmd="python $code_path \
    --results_dirs \
        $bm25_embedder_results_path \
        $bm25_reranker_8b_results_path \
        $bm25_reranker_14b_results_path \
        $bm25_reranker_32b_results_path \
        $embedder_reranker_8b_results_path \
        $embedder_reranker_14b_results_path \
        $embedder_reranker_32b_results_path \
    --weights 14 2 3 8 2 3 8 \
    --norm_weights \
    --merged_model_name bge-reasoner-0928 \
    --top_k 2000 \
    --ori_split_name examples \
    --save_dir $save_dir \
    --norm "

echo "Executing command: $cmd"
eval $cmd
