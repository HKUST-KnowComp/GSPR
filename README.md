# GSPR: Generalizable Safety Policy Reasoners

This repository contains the research implementation for **GSPR: Aligning LLM Safeguards as Generalizable Safety Policy Reasoners**.

GSPR treats a safety taxonomy as part of the model input instead of fixing one taxonomy during training. Given a prompt or prompt-response pair and a list of safety policies, the model produces:

1. a concise reasoning trace inside `<think>...</think>`;
2. a binary decision inside `\safety{safe|unsafe}`; and
3. the most relevant policy inside `\category{...}`.

The training pipeline has two stages:

1. **Cold-start SFT** on policy-level reasoning traces distilled from Gemini 2.5 Flash.
2. **GRPO alignment** with rule-based format, safety-label, and category-label rewards.

## Paper summary

The paper trains on 19 taxonomies containing 167 policies from Aegis, WildGuard, OR-Bench, GUARDSET-X, BeaverTails, and SafeRLHF. Evaluation covers both taxonomies seen during training and unseen taxonomies from OpenAI Moderation, HEx-PHI, T2T, and Do-Not-Answer.

The main reported results are:

| Base model | Evaluation | Safety accuracy | Category accuracy |
| --- | --- | ---: | ---: |
| Qwen2.5-7B-Instruct | In-domain | 85.68 | 78.32 |
| Qwen2.5-7B-Instruct | Out-of-domain | 92.84 | 79.70 |
| Qwen3-8B | In-domain | 86.36 | 77.89 |
| Qwen3-8B | Out-of-domain | 93.11 | 79.85 |

The paper also reports that cold-started GSPR produces substantially shorter explanations than the compared reasoning guardrails: 34.10 average words for the Qwen2.5 model and 77.73 for the Qwen3 model.

## Reproducibility status

**The repository contains the core GSPR implementation, but the current snapshot is not yet a turnkey package for reproducing the paper tables.** See [Known gaps](#known-gaps) before starting an expensive training run.

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

## Hardware

The paper used one node with **8 NVIDIA H800 80 GB GPUs**. The checked-in launchers also assume eight visible GPUs. Smaller hardware will require changes to tensor parallelism, batch sizes, offloading, and possibly sequence length.

## Environment

An exact pinned environment file is not included yet. The code vendors a modified `verl` package and imports at least the following external packages:

```text
torch, transformers, datasets, huggingface_hub, pandas, pyarrow,
ray, hydra-core, omegaconf, tensordict, torchdata, vllm,
flash-attn, peft, tqdm, tabulate, codetiming, psutil
```

Run commands from the repository root so that the vendored `verl` package is importable. Do not assume that the newest releases of these packages are compatible; a pinned `requirements.txt` or environment lock file is still needed for exact reproduction.

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

The paper samples 80 candidates per taxonomy, asks Gemini 2.5 Flash to produce policy-level reasoning, validates the output format and labels, and retains **1,383** examples. SFT uses:

| Setting | Value |
| --- | ---: |
| Optimizer | Adam |
| Learning rate | `1e-5` |
| Warmup ratio | `0.1` |
| Weight decay | `0.01` |
| Global batch size | `16` |
| Epochs | `2` |

`run_sft_verl.sh` reflects these main hyperparameters, but it is a launcher template rather than a ready command. Before running it, provide:

- a training Parquet file whose `extra_info` column contains `input_prompt` and `distilled_response`;
- a validation Parquet file with the same schema;
- a real output directory through `trainer.default_local_dir`;
- the desired base model (`Qwen/Qwen2.5-7B-Instruct` or `Qwen/Qwen3-8B`).

The required final cold-start Parquet file is not included in this snapshot. The checked-in `cold_start_data/sampled_combined.json` has 1,520 pre-distillation candidates and does not contain the two SFT fields above.

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
5. the actor learning rate from the current `5e-7` to the paper's `1e-7` for exact reproduction;
6. the experiment name and logging configuration.

Then launch:

```bash
bash run_training.sh
```

As currently checked in, the launcher starts GRPO from `Qwen/Qwen3-8B`, so it corresponds more closely to the paper's **GSPR without cold start** variant than to the main model.

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

Inference uses vLLM with temperature `0.0` and repetition penalty `1.2`, matching the paper. The current file list also includes ALERT, AttaQ, SorryBench, and other auxiliary evaluations. To reproduce Tables 2 and 3 exactly, aggregate only:

- in-domain: Aegis, WildGuard, SafeRLHF, and BeaverTails;
- out-of-domain: OpenAI Moderation, HEx-PHI, T2T, and Do-Not-Answer.

Analyze generated JSON files with the checked-in analyzer:

```bash
python inference/analyze_safety_predictions_box.py \
  --experiment_name verl-generalize-reasoner/my-experiment \
  --global_step <step>
```

The final command in `run_training.sh` currently references the missing filename `analyze_safety_predictions.py`; use `analyze_safety_predictions_box.py` instead.

## Known gaps

The following items are required for exact end-to-end reproduction but are missing or inconsistent in this repository snapshot:

1. **Cold-start annotation pipeline:** no Gemini 2.5 Flash calling/filtering script is included.
2. **Final cold-start data:** the paper's filtered 1,383-example SFT Parquet file is absent. The included JSON is the 1,520-example candidate pool and lacks SFT targets.
3. **Pinned environment:** there is no `requirements.txt`, Conda environment, container specification, CUDA version, or package lock file.
4. **Launcher paths:** both launchers contain placeholders; their configured checkpoint directories do not currently agree with the paths used by conversion and inference.
5. **GRPO learning rate:** `run_training.sh` uses `5e-7`, while the paper reports `1e-7`.
6. **Cold-start wiring:** `run_training.sh` starts from the untuned Qwen3 base model instead of the Stage 1 checkpoint.
7. **Validation paths:** both SFT and GRPO instantiate validation datasets, but the launchers contain placeholder validation paths.
8. **Evaluation command:** `run_training.sh` references an analyzer filename that is not present.
9. **Exact data snapshots:** dataset revisions/checksums are not pinned, so future downloads may differ from the paper's data.
10. **Closed-source baselines:** scripts/configurations for the paper's Gemini, GPT, and o3 evaluations are not included.
11. **"Others" augmentation:** the templates define an `others` category for WildGuard, but no general augmentation procedure corresponding to the paper's described sampling strategy is visible.
12. **Release metadata:** a license and citation file are not included.

Until these gaps are resolved, the repository can support code inspection and partial experimentation, but it cannot independently reproduce all reported model checkpoints and table values.

## Repository hygiene

Before publishing:

- ignore `__pycache__/` and `*.py[cod]`;
- never commit Hugging Face or Weights & Biases tokens;
- review training examples for personal information and dataset-license constraints;
- decide whether `cold_start_data/` should remain ignored or be released after sanitization and licensing review.

