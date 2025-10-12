#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"

ROOT_DIR=$SCRIPT_DIR/../../..

cd $ROOT_DIR

task_type="short"
eval_root="$ROOT_DIR/bright-evaluation/$task_type"
cache_path="$ROOT_DIR/bright-evaluation/.cache"
reranker_cache_path="$ROOT_DIR/bright-evaluation/.reranker_scores_cache"

dataset_names="theoremqa_theorems"

retrieval_split="reasoner-rewritten-query-0821"
reranking_split="examples"

embedder_path="$eval_root/search_results/$retrieval_split/reason-embed-basic-qwen3-8b-0928"
reranker_path="ljw13/retro-star-qwen3-8b-0928"

################################ mean-score@1 ################################
python main.py \
    --dataset_dir $eval_root/data \
    --output_dir $eval_root/search_results/$retrieval_split \
    --task_type $task_type \
    --eval_name bright_$task_type \
    --dataset_names $dataset_names \
    --retrieval_split $retrieval_split \
    --splits $reranking_split \
    --search_top_k 2000 \
    --overwrite False \
    --k_values 1 10 100 \
    --embedder_name_or_path $embedder_path \
    --embedder_model_class custom \
    --reranker_name_or_path $reranker_path \
    --reranker_model_class sglang-reasoning \
    --rerank_top_k 100 \
    --reranker_sample_k 1 \
    --reranker_batch_size 10240 \
    --reranker_sglang_tp_size 1 \
    --reranker_sglang_dp_size 8 \
    --reranker_sglang_seed 42 \
    --reranker_max_new_tokens 4096 \
    --reranker_enable_thinking False \
    --multiple_pass_reranking_paths '100@1' \
    --reranker_enable_full_parallelism False \
    --cache_path $cache_path \
    --reranker_enable_cache True \
    --reranker_scores_cache_dir $reranker_cache_path

################################ mean-score@16 ################################
python main.py \
    --dataset_dir $eval_root/data \
    --output_dir $eval_root/search_results/$retrieval_split \
    --task_type $task_type \
    --eval_name bright_$task_type \
    --dataset_names $dataset_names \
    --retrieval_split $retrieval_split \
    --splits $reranking_split \
    --search_top_k 2000 \
    --overwrite False \
    --k_values 1 10 100 \
    --embedder_name_or_path $embedder_path \
    --embedder_model_class custom \
    --reranker_name_or_path $reranker_path \
    --reranker_model_class sglang-reasoning \
    --rerank_top_k 30 \
    --reranker_sample_k 16 \
    --reranker_batch_size 512 \
    --reranker_sglang_tp_size 4 \
    --reranker_sglang_dp_size 2 \
    --reranker_sglang_seed 42 \
    --reranker_max_new_tokens 4096 \
    --reranker_enable_thinking False \
    --multiple_pass_reranking_paths  '100@1' '30@16' \
    --reranker_enable_full_parallelism False \
    --cache_path $cache_path \
    --reranker_enable_cache True \
    --reranker_scores_cache_dir $reranker_cache_path
