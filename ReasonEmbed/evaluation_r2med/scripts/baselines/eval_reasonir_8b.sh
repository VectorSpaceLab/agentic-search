#!/bin/bash

# NOTE: set to the true path where the code is located
cd ./ReasonEmbed/evaluation_r2med/code

model_path="reasonir/ReasonIR-8B"

cmd="python main.py \
    --eval_name r2med \
    --use_special_instructions True \
    --dataset_dir ./r2med/data \
    --splits test \
    --corpus_embd_save_dir ./r2med/corpus_embd \
    --output_dir ./r2med/search_results \
    --search_top_k 2000 \
    --cache_path ./cache/data \
    --overwrite False \
    --k_values 1 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./r2med/eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_10 recall_at_100 \
    --embedder_name_or_path $model_path \
    --trust_remote_code True \
    --cache_dir ./cache/model \
    --embedder_batch_size 8 \
    --embedder_query_max_length 512 \
    --embedder_passage_max_length 512 "


echo $cmd
eval $cmd
