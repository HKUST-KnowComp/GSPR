import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#os.environ['HF_HOME'] = "xxxx"
import json
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pandas as pd
# === Import Template Definitions ===
from templates import PROMPT_TPL, RESPONSE_TPL

# def get_prompt_template(test_mode: str, user_prompt: str, response: str = ""):
#     if test_mode == "prompt":
#         return PROMPT_TPL["RSafe_Prompt"].format(prompt=user_prompt.strip())
#     else:  # test_mode == "response"
#         return RESPONSE_TPL["RSafe_Response"].format(prompt=user_prompt.strip(), response=response.strip())

def qwen_prompt_dataset(dataset: list, sampling_params: SamplingParams, tokenizer, llm, test_mode: str = "prompt", args = None) -> list:
    texts = []
    #for sample in dataset[:10]:
    for sample in dataset:
        if test_mode in ["prompt", "both"]:
            messages = sample.get("prompt", "")
            
            if args.experiment_name == "verl-generalize-reasoner/wildguard_baseline":
                FORMAT = """<s><|user|>
[INST] {prompt}

---

Answers: [/INST]
</s>"""
                text = FORMAT.format( prompt= messages[0]['content'])
            elif args.experiment_name == "verl-generalize-reasoner/shieldgemma_baseline":
                text = messages[0]['content'] + "\n\n### Response:\n"
            else:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)

        if test_mode in ["response", "both"]:
            messages = sample.get("prompt", "")
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)

    outputs = llm.generate(texts, sampling_params)
    return [output.outputs[0].text for output in outputs]

def run_inference(prompt_files: list, response_files: list, output_dir: str, sampling_params: SamplingParams, tokenizer, llm, test_mode: str = "prompt", args = None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if test_mode in ["prompt", "both"]:
        for input_file in prompt_files:
            # with open(input_file, "r", encoding="utf-8") as f:
            #     dataset = json.load(f)
            dataset = parquet_reader(input_file)
            responses = qwen_prompt_dataset(dataset, sampling_params, tokenizer, llm, "prompt", args = args)
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_path = os.path.join(output_dir, f"{base_name}_prompt_results.json")
            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump([{ "prompt": s.get("prompt", "")[0]['content'], 
                            "model_assessment": r,
                             "safety":  s["reward_model"]["ground_truth"],
                             'category': s["extra_info"]['category'],
                             'category_dict': s["extra_info"]['category_dict']} for s, r in zip(dataset, responses)], out_f, ensure_ascii=False, indent=2)
            print(f"Prompt results saved to {output_path}")

    if test_mode in ["response", "both"]:
        for input_file in response_files:
            # with open(input_file, "r", encoding="utf-8") as f:
            #     dataset = json.load(f)
            dataset = parquet_reader(input_file)
            responses = qwen_prompt_dataset(dataset, sampling_params, tokenizer, llm, "response",  args = args)
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_path = os.path.join(output_dir, f"{base_name}_response_results.json")
            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump([{ "prompt": s.get("prompt", "")[0]['content'], 
                            #"response": s.get("response", ""), 
                            "model_assessment": r,
                             "safety":  s["reward_model"]["ground_truth"],
                             'category': s["extra_info"]['category'],
                             'category_dict': s["extra_info"]['category_dict'] } for s, r in zip(dataset, responses)], out_f, ensure_ascii=False, indent=2)
            print(f"Response results saved to {output_path}")


def parquet_reader(file_path: str):
    df = pd.read_parquet(file_path)
    # Convert to a list of dictionaries (row-wise)
    data_dict = df.to_dict(orient="records")
    return data_dict
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', type=str, default="xxxx/generalizable_safety", help='Base local directory')
    parser.add_argument('--huggingface_name', type=str, default="", help='HF model name')
    parser.add_argument('--experiment_name', type=str, default="verl-generalize-reasoner/Qwen2.5-7B-Instruct_baseline")
    #parser.add_argument('--experiment_name', type=str, default="verl-generalize-reasoner/wildguard_baseline")
    parser.add_argument('--global_step', type=int, default=766)
    parser.add_argument('--base_model_name', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--test_mode', type=str, choices=["prompt", "response", "both"], default="prompt")
    parser.add_argument('--sft', action='store_true')
    args = parser.parse_args()

    prompt_input_files = [
        "datasets_test_box/aegis_prompt_test.parquet",
        "datasets_test_box/wildguardmix_prompt_test.parquet",
        "datasets_test_box/openai_prompt_test.parquet", 
        "datasets_test_box/BeaverTails_prompt_test.parquet", 
        "datasets_test_box/PKU-SafeRLHF_default_0_prompt_test.parquet", 
        "datasets_test_box/alert_adv_prompt_test.parquet",
        "datasets_test_box/alert_prompt_test.parquet",
        "datasets_test_box/attaq_prompt_test.parquet",
        "datasets_test_box/hex_phi_prompt_test.parquet",
        "datasets_test_box/sorry_prompt_test.parquet",
        "datasets_test_box/T2T_prompt_test.parquet",
        "datasets_test_box/do_not_answer_prompt_test.parquet",
    ]

    response_input_files = [
        # "../data/saferlhf/RLHF_merged.json",
        # "../data/beavertails/BeaverTails.json",
        # "../data/xstest/xstest_merged.json",
        # "../data/wildguardtest/wildguardtest_response.json",
    ]

    prompt_input_files = [os.path.join(args.local_dir, file) for file in prompt_input_files]
    response_input_files = [os.path.join(args.local_dir, file) for file in response_input_files]
    if args.remote:
        model_path = args.huggingface_name
    elif args.sft:
        model_path = f"{args.local_dir}/verl_sft/{args.experiment_name}/checkpoint-{args.global_step}"
    else:
        model_path = f"{args.local_dir}/models/{args.experiment_name}/global_step_{args.global_step}/huggingface"

    #overwrite to test qwen2.5-7b-instruct
    if args.experiment_name == "verl-generalize-reasoner/Qwen2.5-7B-Instruct_baseline":
        model_path = 'Qwen/Qwen2.5-7B-Instruct'
        print(f'Using the baseline model: {model_path}')
    elif args.experiment_name == "verl-generalize-reasoner/Qwen3-8B_baseline":
        model_path = 'Qwen/Qwen3-8B'
        print(f'Using the baseline model: {model_path}')
    elif args.experiment_name == "verl-generalize-reasoner/Qwen2.5-7B_cold_start_86":
        model_path = 'models/verl-generalize-reasoner/Qwen2.5-7B-Instruct_cold_start/global_step_86'
        print(f'Using the baseline model: {model_path}')
    elif args.experiment_name == "verl-generalize-reasoner/Qwen3-8B_cold_start_86":
        model_path = 'models/verl-generalize-reasoner/Qwen3-8B_cold_start/global_step_86'
        print(f'Using the baseline model: {model_path}')
    elif args.experiment_name == "verl-generalize-reasoner/Qwen2.5-7B_cold_start_172":
        model_path = 'models/verl-generalize-reasoner/Qwen2.5-7B-Instruct_cold_start/global_step_172'
        print(f'Using the baseline model: {model_path}')
    elif args.experiment_name == "verl-generalize-reasoner/Qwen3-8B_cold_start_172":
        model_path = 'models/verl-generalize-reasoner/Qwen3-8B_cold_start/global_step_172'
        print(f'Using the baseline model: {model_path}')

    elif args.experiment_name == "verl-generalize-reasoner/wildguard_baseline":
        model_path = 'allenai/wildguard'
        print(f'Using the baseline model: {model_path}')
    elif args.experiment_name == "verl-generalize-reasoner/shieldgemma_baseline":
        model_path = 'google/shieldgemma-9b'
        print(f'Using the baseline model: {model_path}')
    elif args.experiment_name == "verl-generalize-reasoner/GuardReasoner_baseline":
        model_path = 'yueliu1999/GuardReasoner-8B'
        print(f'Using the baseline model: {model_path}')
    elif args.experiment_name == "verl-generalize-reasoner/Llama-Guard_baseline":
        model_path = 'meta-llama/Llama-Guard-3-8B'
        print(f'Using the baseline model: {model_path}')
    output_dir = os.path.join(args.local_dir, f"result/{args.experiment_name}_step_{args.global_step}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    sampling_params = SamplingParams(temperature=0.0, top_p=0.8, repetition_penalty=1.2, max_tokens=2048)

    llm = LLM(model=model_path, gpu_memory_utilization=0.85) # Use 90% of GPU memory
    run_inference(prompt_input_files, response_input_files, output_dir, sampling_params, tokenizer, llm, args.test_mode, args = args)