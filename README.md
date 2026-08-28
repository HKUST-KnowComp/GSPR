# GSPR: Generalizable Safety Policy Reasoners

This repository contains the research implementation for **GSPR: Aligning LLM Safeguards as Generalizable Safety Policy Reasoners** (https://arxiv.org/abs/2509.24418).

GSPR treats a safety taxonomy as part of the model input instead of fixing one taxonomy during training. Given a prompt or prompt-response pair and a list of safety policies, the model produces:

1. a concise reasoning trace inside `<think>...</think>`;
2. a binary decision inside `\safety{safe|unsafe}`; and
3. the most relevant policy inside `\category{...}`.

The training pipeline has two stages:

1. **Cold-start SFT** on policy-level reasoning traces distilled from Gemini 2.5 Flash.
2. **GRPO alignment** with rule-based format, safety-label, and category-label rewards.

## Paper summary

The paper trains on 19 taxonomies containing 167 policies from Aegis, WildGuard, OR-Bench, GUARDSET-X, BeaverTails, and SafeRLHF. Evaluation covers both taxonomies seen during training and unseen taxonomies from OpenAI Moderation, HEx-PHI, T2T, and Do-Not-Answer.

## Reproducibility status

**The repository contains the core GSPR implementation.** 

Implemented components include:

- flexible prompt and taxonomy construction;
- prompt-safety and response-safety dataset adapters;
- FSDP supervised fine-tuning;
- VERL-based GRPO training;
- the paper's format, repetition, language, safety, and category rewards;
- FSDP-to-Hugging-Face checkpoint conversion;
- vLLM inference and safety/category metric calculation.

## Repository layout

```text
.
├── get_data.py                         # Prompt-safety data preparation
├── response_code/
│   └── get_response_data.py            # Response-safety data preparation
├── templates.py                        # Prompt-safety taxonomies/templates
├── response_code/response_templates.py # Response-safety taxonomies/templates
├── label_fn.py                         # Prompt-safety label adapters
├── response_code/label_response_fn.py  # Response-safety label adapters
├── cold_start_data/                    # Candidate cold-start data utilities
├── run_sft_verl.sh                     # Cold-start SFT launcher template
├── run_training.sh                     # GRPO launcher template
├── verl/                               # Vendored VERL 0.2.0.dev implementation
├── merge_fsdp_to_hf.py                 # Merge GRPO FSDP checkpoints
└── inference/
    ├── inference_trained_model.py       # vLLM inference
    └── analyze_safety_predictions_box.py
```



## Environment

An exact pinned environment file is not included yet. The code vendors a modified `verl` package and imports at least the following external packages:

```text
torch, transformers, datasets, huggingface_hub, pandas, pyarrow,
ray, hydra-core, omegaconf, tensordict, torchdata, vllm,
flash-attn, peft, tqdm, tabulate, codetiming, psutil
```

Run commands from the repository root so that the vendored `verl` package is importable. The newest releases of the verl packages may not be compatible (vLLM version mismatch problem). 


Keep credentials outside the repository:

```bash
export HF_TOKEN="your-hugging-face-token"
export WANDB_API_KEY="your-wandb-key"  # only if using Weights & Biases
export HF_HOME="/path/to/huggingface/cache"
```

Do not replace the placeholder strings in the launchers with real keys before committing.

## Data preparation

The paper samples up to 3,000 safe and 3,000 unsafe training examples from each split. Test sets use up to 1,500 examples per class.

### 1. Prompt-safety datasets

Training data:

```bash
python get_data.py \
  --local_dir datasets \
  --split train \
  --num_safe 3000 \
  --num_unsafe 3000 \
  --data_source checklist_reward \
  --HF_token "$HF_TOKEN"
```

Evaluation data:

```bash
python get_data.py \
  --local_dir datasets_test_box \
  --split test \
  --num_safe 1500 \
  --num_unsafe 1500 \
  --data_source checklist_reward \
  --HF_token "$HF_TOKEN"
```

### 2. Response-safety datasets

The response-data script currently constructs its output path relative to `response_code/`, so run it from that directory:

```bash
(
  cd response_code
  python get_response_data.py \
    --local_dir datasets \
    --split train \
    --num_safe 3000 \
    --num_unsafe 3000 \
    --data_source checklist_reward \
    --HF_token "$HF_TOKEN"
)
```

Evaluation data:

```bash
(
  cd response_code
  python get_response_data.py \
    --local_dir datasets_test_box \
    --split test \
    --num_safe 1500 \
    --num_unsafe 1500 \
    --data_source checklist_reward \
    --HF_token "$HF_TOKEN"
)
```

These commands download third-party datasets. Review their current licenses and access requirements before redistribution.

## Stage 1: cold-start supervised fine-tuning

The paper samples 80 candidates per taxonomy, asks Gemini 2.5 Flash to produce policy-level reasoning, validates the output format and labels, and retains **1,383** examples. 

`run_sft_verl.sh` reflects these main hyperparameters. Before running it, provide:

- a training Parquet file whose `extra_info` column contains `input_prompt` and `distilled_response`;
- a real output directory through `trainer.default_local_dir`;
- the desired base model (`Qwen/Qwen2.5-7B-Instruct` or `Qwen/Qwen3-8B`).


After supplying those artifacts and replacing the path placeholders, launch with:

```bash
bash run_sft_verl.sh
```

## Stage 2: GRPO alignment

The paper's GRPO configuration is:

| Setting | Value |
| --- | ---: |
| GPUs | 8 x H800 80 GB |
| Global batch size | `128` |
| PPO mini-batch size | `64` |
| Micro-batch size per GPU | `8` |
| Learning rate | `1e-7` |
| Epochs | `1` |
| Rollouts per prompt | `5` |
| Maximum response length | `1024` |
| Temperature | `0.7` |
| Top-p | `0.8` |
| Repetition penalty | `1.2` |
| Safety/category reward weights | `0.55 / 0.45` |

The reward implementation is in `verl/utils/reward_score/safety_checklist_box.py`. It checks the required output structure, rejects Chinese-language mixing and repeated 5-grams, applies the paper's overlong-response penalty, and combines safety and category correctness with weights 0.55 and 0.45.

Before using `run_training.sh`, update all of the following:

1. `BASE_DIR` and `HF_HOME`;
2. `data.val_files`, which is currently a placeholder;
3. `trainer.default_local_dir`, so it matches the checkpoint path expected later in the script;
4. `BASE_MODEL`, which must point to the Stage 1 checkpoint for the paper's **GSPR with cold start** result;
5. the experiment name and logging configuration.

Then launch:

```bash
bash run_training.sh
```

The launcher starts GRPO from `Qwen/Qwen3-8B`, so it corresponds to the paper's **GSPR without cold start** variant.

## Checkpoint conversion

GRPO saves sharded FSDP checkpoints under:

```text
<model-root>/<experiment>/global_step_<step>/actor/
```

Merge one checkpoint into Hugging Face format with:

```bash
python merge_fsdp_to_hf.py \
  --local_dir /path/to/model-root \
  --experiment_name verl-generalize-reasoner/my-experiment \
  --global_step <step> \
  --base_model_name Qwen/Qwen3-8B
```

The merged model is written to `global_step_<step>/huggingface/`.

## Inference and evaluation

`inference/inference_trained_model.py` expects evaluation Parquet files under:

```text
<local-dir>/datasets_test_box/
```

Example:

```bash
python inference/inference_trained_model.py \
  --local_dir /path/to/project-root \
  --experiment_name verl-generalize-reasoner/my-experiment \
  --global_step <step> \
  --base_model_name Qwen/Qwen3-8B
```

Inference uses vLLM with temperature `0.0` and repetition penalty `1.2`. The current file list also includes ALERT, AttaQ, SorryBench, and other auxiliary evaluations. To reproduce Tables 2 and 3 exactly, aggregate only:

- in-domain: Aegis, WildGuard, SafeRLHF, and BeaverTails;
- out-of-domain: OpenAI Moderation, HEx-PHI, T2T, and Do-Not-Answer.

Analyze generated JSON files with the checked-in analyzer:

```bash
python inference/analyze_safety_predictions_box.py \
  --experiment_name verl-generalize-reasoner/my-experiment \
  --global_step <step>
```

The final command in `run_training.sh` currently references the missing filename `analyze_safety_predictions.py`; use `analyze_safety_predictions_box.py` instead.



## Citation
Please kindly cite our paper if you found our method and resources helpful!

## Miscellaneous
Please send any questions about the code and/or the method to hlibt@connect.ust.hk.
<div align="center">

