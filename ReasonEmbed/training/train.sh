#!/bin/bash

# set huggingface mirror
export HF_ENDPOINT=https://hf-mirror.com

export WANDB_MODE=disabled

output_dir="YOUR_OUTPUT_DIR"

mkdir -p $output_dir


model_name_or_path="${MODEL_NAME_OR_PATH:-hanhainebula/qwen3-8b-ft-msmarco}"
train_data="${TRAIN_DATA:-./data/reason-embed-data/reason-embed-data-0928}"
cache_dir="${CACHE_DIR:-${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}}"
cache_path="${CACHE_PATH:-${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}}"


# set large epochs and small batch size for testing
num_train_epochs=1
per_device_train_batch_size=8

# set num_gpus to 2 for testing
num_gpus=8

model_args="\
    --model_name_or_path $model_name_or_path \
    --cache_dir $cache_dir \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
    --save_merged_lora_model True \
"

data_args="\
    --train_data $train_data \
    --cache_path $cache_path \
    --train_group_size 16 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_format 'Instruct: {}\nQuery: {}' \
    --knowledge_distillation False \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
"

training_args="\
    --output_dir $output_dir \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --bf16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ./ds_stage1.json \
    --logging_steps 1 \
    --save_steps 500 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --sub_batch_size 4 \
    --qri_start_step 100 \
"

cmd="torchrun --nproc_per_node $num_gpus \
    main.py \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd 2>&1 | tee ${output_dir}/train.log
