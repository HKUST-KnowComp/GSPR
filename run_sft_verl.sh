#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export WANDB_API_KEY="xxxxxxxxx"
export HF_HOME="xxxxxxxxx"
export TRITON_CACHE_DIR='.triton_cache'

BASE_DIR="xxxxxxxxx"
#BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL="Qwen/Qwen3-8B"
project_name="verl-generalize-reasoner"
experiment_name="Qwen3-8B_cold_start"
MODEL_DIR=".model_save/${project_name}/${experiment_name}"


train_files='xxxxxxxxx'
test_files='xxxxxxxxx'

echo "Training files: $train_files"
echo "Testing files: $test_files"
echo "Base model: $BASE_MODEL"

torchrun --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$train_files" \
    data.val_files="$test_files"  \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['input_prompt'] \
    +data.response_dict_keys=['distilled_response'] \
    data.max_length=2048 \
    data.truncation=right \
    data.micro_batch_size_per_gpu=2 \
    data.train_batch_size=16 \
    model.partial_pretrain=$BASE_MODEL \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=2 \
    trainer.logger='["console","wandb"]' \
    "$@" 2>&1 | tee "safety_prompt_training_${experiment_name}.log"
