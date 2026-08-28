#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export WANDB_API_KEY="xxxxxxxxx"
export HF_HOME="xxxxxxxxx"
BASE_DIR="xxxxxxxxx"
# Configuration
DATA_DIR="${BASE_DIR}/datasets"

# Dataset names
DATASETS=("wildguardmix" "aegis" "or_bench_hard1k" "or_bench_toxic")
DATASETS+=("polyguard_edu_ai4edu" "polyguard_edu_college")
DATASETS+=("polyguard_hr_adobe" "polyguard_hr_amazon" "polyguard_hr_apple" "polyguard_hr_bytedance" "polyguard_hr_google" "polyguard_hr_ibm" "polyguard_hr_intel" "polyguard_hr_meta" "polyguard_hr_microsoft" "polyguard_hr_nvidia")
DATASETS+=("PKU-SafeRLHF_default_0" "PKU-SafeRLHF_default_1" "BeaverTails")

RESPONSE_DATASETS=("Beavertails", "pku_saferlhf")

# Dataset configuration flags
ENABLE_ORIGINAL_BALANCED=true
ENABLE_RESPONSE_BALANCED=true


#BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL="Qwen/Qwen3-8B"
project_name="verl-generalize-reasoner"
experiment_name="temp_trail_qwen3_epoch1_response_update_checklist_reward"
MODEL_DIR="${BASE_DIR}/models/${project_name}/${experiment_name}"
iteration_file="${MODEL_DIR}/latest_checkpointed_iteration.txt"
# Initialize file lists
train_files="["
test_files="["
epoch=1


for source in "${DATASETS[@]}"; do
        train_files+="'$DATA_DIR/${source}_prompt_train.parquet', "
        #test_files+="'$DATA_DIR/${source}_prompt_test.parquet', "
done


# for source in "${RESPONSE_DATASETS[@]}"; do
#         train_files+="'$DATA_DIR/${source}_response_train.parquet', "
#         test_files+="'$DATA_DIR/${source}_response_test.parquet', "
# done


#data.train_files="$train_files" \
#data.val_files="$test_files" \

# Close file lists
train_files=${train_files%, }"]"
test_files=${test_files%, }"]"

echo "Training files: $train_files"
echo "MODEL_DIR: $MODEL_DIR"
echo "Base model: $BASE_MODEL"
# Training configuration
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.repetition_penalty=1.2 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.repetition_penalty=1.2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    data.train_files="$train_files" \
    data.val_files="xxxxxxxxx" \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.truncation='right' \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=4 \
    data.train_batch_size=128 \
    actor_rollout_ref.model.path="$BASE_MODEL"\
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    trainer.logger=['wandb'] \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=1000 \
    trainer.total_epochs="$epoch"  \
    trainer.log_val_generations=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    trainer.n_gpus_per_node=8  \
    trainer.nnodes=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    "$@" 2>&1 | tee "safety_prompt_training_${experiment_name}.log"


# Save the latest iteration number
step_number=$(cat "$iteration_file")


### convert the FSDP checkpoint to HF format
python3 merge_fsdp_to_hf.py \
    --local_dir "${BASE_DIR}/models" \
    --experiment_name "${project_name}/${experiment_name}" \
    --global_step "$step_number" \
    --base_model_name "$BASE_MODEL"


### Run inference on the trained model
cd "${BASE_DIR}/inference"
python3 inference_trained_model.py \
    --local_dir "${BASE_DIR}" \
    --experiment_name "${project_name}/${experiment_name}" \
    --global_step "$step_number" \
    --base_model_name "$BASE_MODEL"


### Get the evaluation results
cd "${BASE_DIR}/inference"
python3 analyze_safety_predictions.py \
    --experiment_name "${project_name}/${experiment_name}" \
    --global_step "$step_number"

