#!/usr/bin/env python
# Combined script to generate response classification datasets

import os
import pandas as pd
import argparse
import json
from datasets import load_dataset, concatenate_datasets
from tqdm.auto import tqdm # For displaying progress bar
from transformers import AutoTokenizer
from label_fn import get_label_fn # Custom function to get data labels
import templates # Custom templates for formatting input prompts
from huggingface_hub import login
from pprint import pprint
import re

# Random number generator seed to ensure reproducibility
RNG = 50

def load_tokenizer():
    """Load the Qwen2.5-7b tokenizer."""
    print("Loading Qwen2.5-7b tokenizer...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, # Use the tokenizer corresponding to the model to be trained (here, the model to be trained later is Qwen2.5-7B-Instruct)
        trust_remote_code=True, # Allow loading of remote code (since tokenization code for some models may be included in their own model repositories, not built-in code in the Hugging Face transformers library, requiring remote loading)
        use_fast=True # Fast tokenizer is implemented based on Rust; compared to pure Python tokenizers, it offers significantly faster processing speed (most models support fast tokenizers)
    )
    return tokenizer


def process_prompt_dataset(dataset_name, subset, prompt_key, label_fn, tokenizer, max_length, dataset_tag,
                           split = 'train', sample_ratio = 1.0, args = None):
    '''
    label_fn returns a dictionary with:
        safety: "safe" or "unsafe"
        category: category of the item (dataset dependent), e.g., "harassment", "hate", etc.
    args:
        dataset_name: name of the dataset, e.g., "bench-llm/or-bench"
        subset: subset of the dataset, e.g., "or-bench-hard-1k"
        split: split of the dataset, e.g., "train", "test", etc.
        prompt_key: key in the dataset item that contains the prompt text
        label_fn: function to get the label from the item
        tokenizer: tokenizer to format the prompt
        max_length: maximum length of the tokenized prompt
        dataset_tag: tag for the dataset, used for output naming
        sample_ratio: ratio of samples to keep, default is 1.0 (keep all)
    Returns:
        DataFrame with columns ['prompt', 'answer', 'category']
    '''

    # Load dataset: Specify which train/test split of the dataset to use
    if isinstance(split, list):
        if dataset_name == "LLM-Tuning-Safety/HEx-PHI":
            hex_phi_data_files = {
                "Category_1_Illegal_Activity": "hf://datasets/LLM-Tuning-Safety/HEx-PHI/category_1.csv",
                "Category_3_Hate_Harass_Violence": "hf://datasets/LLM-Tuning-Safety/HEx-PHI/category_3.csv",
                "Category_4_Malware": "hf://datasets/LLM-Tuning-Safety/HEx-PHI/category_4.csv",
                "Category_5_Physical_Harm": "hf://datasets/LLM-Tuning-Safety/HEx-PHI/category_5.csv",
                "Category_6_Economic_Harm": "hf://datasets/LLM-Tuning-Safety/HEx-PHI/category_6.csv",
                "Category_7_Fraud_Deception": "hf://datasets/LLM-Tuning-Safety/HEx-PHI/category_7.csv",
                "Category_8_Adult_Content": "hf://datasets/LLM-Tuning-Safety/HEx-PHI/category_8.csv",
                "Category_9_Political_Campaigning": "hf://datasets/LLM-Tuning-Safety/HEx-PHI/category_9.csv",
                "Category_10_Privacy_Violation_Activity": "hf://datasets/LLM-Tuning-Safety/HEx-PHI/category_10.csv",
                "Category_11_Tailored_Financial_Advice": "hf://datasets/LLM-Tuning-Safety/HEx-PHI/category_11.csv"
            }
            dataset_dict = load_dataset(
                "csv",  
                data_files=hex_phi_data_files,  
                column_names=["prompt"],  
                skiprows=1  
            )
        else:
            dataset_dict = load_dataset(dataset_name, subset) if subset else load_dataset(dataset_name)
        
        if dataset_name == "AI-Secure/PolyGuard":
            all_splits_with_safety = []
            for split_name in split:
                if split_name not in dataset_dict:
                    raise KeyError(f"Split '{split_name}' does not exist! Available splits: {list(dataset_dict.keys())}")
                split_dataset = dataset_dict[split_name]
                if "unsafe" in split_name:
                    safety_label = "unsafe"
                elif "safe" in split_name:
                    safety_label = "safe"
                else:
                    safety_label = "unknown"  # Or raise ValueError(f"Split name {split_name} has no matching safety label")
                # Add 'safety' column to all samples in the current split
                split_dataset = split_dataset.add_column(
                    "safety", 
                    [safety_label] * len(split_dataset)  # Fill safety label for each sample
                )
                all_splits_with_safety.append(split_dataset)
            dataset = concatenate_datasets(all_splits_with_safety)
        elif dataset_name == "LLM-Tuning-Safety/HEx-PHI":
            all_splits_with_category = []
            for split_name in split:
                if split_name not in dataset_dict:
                    raise KeyError(f"Split '{split_name}' does not exist! Available splits: {list(dataset_dict.keys())}")
                split_dataset = dataset_dict[split_name]
                pattern = r'Category_\d+_(.+)'
                match = re.search(pattern, split_name)
                if match:
                    category_label = match.group(1).lower()
                else:
                    category_label = "unknown"
                split_dataset = split_dataset.add_column(
                    "category", 
                    [category_label] * len(split_dataset)  # Fill category label for each sample
                )
                all_splits_with_category.append(split_dataset)
            dataset = concatenate_datasets(all_splits_with_category)
    else:
        dataset = load_dataset(dataset_name, subset)[split] if subset else load_dataset(dataset_name)[split]

    data, skipped = [], 0 # 'data' stores qualified processed data; 'skipped' counts samples skipped due to excessive length

    # Iterate through the dataset with progress bar
    for item in tqdm(dataset, desc=f"Processing {dataset_tag}"):
        # Extract prompt text
        if dataset_name == 'Babelscape/ALERT':
            raw_prompt = item[prompt_key]
            pattern = r'### Instruction:\s*(.*?)\s*### Response:'
            match = re.search(pattern, raw_prompt, re.DOTALL)
            prompt = match.group(1).strip()
        elif dataset_name == 'sorry-bench/sorry-bench-202406':
            prompt = item[prompt_key][0]
        elif dataset_name == "LLM-Tuning-Safety/HEx-PHI":
            prompt_key = "prompt"
            prompt = item[prompt_key]
        else:
            prompt = item[prompt_key]

        # Extract label
        try:
            label = label_fn(item)
        except Exception as e:
            print(f"Error processing item: {item}, error: {e}")
            continue
        
        # Extract safety label and category, stripping leading/trailing spaces
        safety = label['safety'].strip()
        category = label.get('category', None).strip()

        ### Skip multi-label samples for Aegis dataset
        # When processing the nvidia/Aegis-AI-Content-Safety-Dataset-2.0, filter out items belonging to multiple categories, keeping only those with a single category
        if dataset_name in ['nvidia/Aegis-AI-Content-Safety-Dataset-2.0', 'mmathys/openai-moderation-api-evaluation'] :
            category_list = category.split(',') # Split category into a list by comma
            if len(category_list) > 1:
                continue

        # Format input prompt and category dictionary using templates
        chat_prompt, category_dict_rt = templates.format_input_prompt(prompt, label, dataset_name, subset, dataset_tag, split = None, sample_ratio = sample_ratio, args = args)
        # Apply chat template to format the prompt
        formatted = tokenizer.apply_chat_template([
            {"role": "user", "content": chat_prompt}
        ], tokenize=False, add_generation_prompt=True)
        # Check length and keep only samples not exceeding max_length
        if len(tokenizer.encode(formatted)) <= max_length: # Tokenize and encode to get a sequence of token IDs (i.e., a list of integers)
            data.append({"prompt": prompt,
                         "answer": safety,
                         "category": category,
                         "chat_prompt": chat_prompt,
                         "category_dict": category_dict_rt})
        else:
            skipped += 1
    print(f"Kept {len(data)} samples, skipped {skipped} too long")
    return pd.DataFrame(data)

def balance_dataset(df, num_safe=None, num_unsafe=None, allow_replace=False):
    """
        Balance the number of safe and unsafe samples in the dataset

        Args:
            df: Original dataset
            num_safe: Target number of safe samples
            num_unsafe: Target number of unsafe samples
            allow_replace: Whether to allow sampling with replacement (when target number exceeds actual number)

        Returns:
            DataFrame: Balanced dataset
    """

    # Separate safe and unsafe samples
    df_safe = df[df['answer'] == 'safe']
    df_unsafe = df[df['answer'] == 'unsafe']

    # If sampling without replacement, ensure target number does not exceed actual sample count
    if not allow_replace:
        num_safe = min(num_safe, len(df_safe)) if num_safe is not None else len(df_safe)
        num_unsafe = min(num_unsafe, len(df_unsafe)) if num_unsafe is not None else len(df_unsafe)

    # Sample to reach target numbers
    df_safe = df_safe.sample(n=num_safe, replace=allow_replace, random_state=RNG)
    df_unsafe = df_unsafe.sample(n=num_unsafe, replace=allow_replace, random_state=RNG)

    # Merge and shuffle the order
    df_balanced = pd.concat([df_safe, df_unsafe]).sample(frac=1, random_state=RNG).reset_index(drop=True)
    return df_balanced
    
def format_prompt_dataset(df, dataset_tag, split, data_source):
    """
        Format the dataset into a specific structure for subsequent training or evaluation

        Args:
            df: Processed dataset
            dataset_tag: Dataset tag

        Returns:
            DataFrame: Formatted dataset
    """
    output = []
    # Iterate through the dataset with progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Formatting {dataset_tag}"):
        #chat_prompt = convert_prompt_template(row['prompt'], wo_category=wo_category)
        chat_prompt = row["chat_prompt"]
        category_dict = row["category_dict"]

        # Construct output structure
        output.append({
            "data_source": data_source, # Data source identifier (standard answer checklist, usable for training reward models)
            "prompt": [{"role": "user", "content": chat_prompt}], # Formatted prompt
            "ability": "safety", # Type of ability to evaluate
            "reward_model": {
                "style": "rule", # Reward model type
                "ground_truth": row['answer'] # Ground truth safety label (safe or unsafe)
            },
            "extra_info": {
                "split": split, # Data split type
                "dataset": dataset_tag, # Dataset tag
                "original_prompt": row['prompt'], # Original user prompt
                "category": row['category'], # Actual safety category
                ### Convert to JSON format to avoid pyarrow errors (schema mismatch if it is a dict across multiple train data files),
                "category_dict": json.dumps(category_dict) # Convert category dictionary to JSON string to avoid formatting issues in subsequent processing
            }
        })
    return pd.DataFrame(output)

def prepare_all_prompt_datasets(local_dir, max_length, num_safe, num_unsafe, split='train',data_source='checklist_reward', args=None):
    """
        Prepare all configured prompt datasets, process them, and save to local storage

        Args:
            local_dir: Local save directory
            max_length: Maximum token length
            num_safe: Number of safe samples
            num_unsafe: Number of unsafe samples
            split: Data split type (train/test)
    """
    tokenizer = load_tokenizer()
    os.makedirs(local_dir, exist_ok=True)  # Create save directory if it does not exist

    ## Dataset source, subset, key for the prompt, split
    # Dataset configuration: Name -> (Data source, Subset, Prompt key, Split)
    configs = {
        'aegis': ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", None, "prompt",split),
        'polyguard_edu_college':(
            "AI-Secure/PolyGuard",  # Dataset source
            "education",             # Subset
            "instance",              # Prompt field name
            [                        # Split: List of 2 split names to merge
                "COLLEGE_BOARD_AP_safe", 
                "COLLEGE_BOARD_AP_unsafe", 
            ]
        ),
        'polyguard_edu_ai4edu':(
            "AI-Secure/PolyGuard",  
            "education",             
            "instance",             
            [                       
                "AI_FOR_EDUCATION_safe", 
                "AI_FOR_EDUCATION_unsafe"
            ]
        ),
        # Google-related splits
        'polyguard_hr_google': (
            "AI-Secure/PolyGuard", 
            "hr", 
            "instance", 
            [                        
                "Google_safe",  
                "Google_unsafe"
            ]
        ),
        # Microsoft-related splits
        'polyguard_hr_microsoft': (
            "AI-Secure/PolyGuard", 
            "hr", 
            "instance", 
            [                        
                "Microsoft_safe",  
                "Microsoft_unsafe"
            ]
        ),
        # Amazon-related splits
        'polyguard_hr_amazon': (
            "AI-Secure/PolyGuard", 
            "hr", 
            "instance", 
            [                        
                "Amazon_safe",  
                "Amazon_unsafe"
            ]
        ),
        # Apple-related splits
        'polyguard_hr_apple': (
            "AI-Secure/PolyGuard", 
            "hr", 
            "instance", 
            [                        
                "Apple_safe",  
                "Apple_unsafe"
            ]
        ),
        # Meta-related splits
        'polyguard_hr_meta': (
            "AI-Secure/PolyGuard", 
            "hr", 
            "instance", 
            [                        
                "Meta_safe",  
                "Meta_unsafe"
            ]
        ),
        # NVIDIA-related splits
        'polyguard_hr_nvidia': (
            "AI-Secure/PolyGuard", 
            "hr", 
            "instance", 
            [                        
                "NVIDIA_safe",  
                "NVIDIA_unsafe"
            ]
        ),
        # IBM-related splits
        'polyguard_hr_ibm': (
            "AI-Secure/PolyGuard", 
            "hr", 
            "instance", 
            [                        
                "IBM_safe",  
                "IBM_unsafe"
            ]
        ),
        # Intel-related splits
        'polyguard_hr_intel': (
            "AI-Secure/PolyGuard", 
            "hr", 
            "instance", 
            [                        
                "Intel_safe",  
                "Intel_unsafe"
            ]
        ),
        # Adobe-related splits
        'polyguard_hr_adobe': (
            "AI-Secure/PolyGuard", 
            "hr", 
            "instance", 
            [                        
                "Adobe_safe",  
                "Adobe_unsafe"
            ]
        ),
        # ByteDance-related splits (Note: The last split in the original list was incomplete in your screenshot; the full name is assumed here and can be adjusted based on actual conditions)
        'polyguard_hr_bytedance': (
            "AI-Secure/PolyGuard", 
            "hr", 
            "instance", 
            [                        
                "ByteDance_safe",  
                "ByteDance_unsafe"  
            ]
        )
    }
    # Add different datasets based on split type
    if split == 'train':
        configs['wildguardmix'] = ('allenai/wildguardmix','wildguardtrain', "prompt", split)
        configs['or_bench_hard1k'] = ("bench-llm/or-bench", "or-bench-hard-1k", "prompt", split)
        configs['or_bench_toxic'] = ("bench-llm/or-bench", "or-bench-toxic", "prompt", split)

    elif split == 'test':
        configs['wildguardmix'] = ('allenai/wildguardmix','wildguardtest', "prompt", split)
        configs['alert'] = ('Babelscape/ALERT','alert',"prompt",split)
        configs['alert_adv'] = ('Babelscape/ALERT','alert_adversarial',"prompt",split)
        configs['sorry'] = ('sorry-bench/sorry-bench-202406',None,"turns",'train') # thoungh the split_type is 'train', use it as 'test'
        configs['hex_phi'] = ('LLM-Tuning-Safety/HEx-PHI',None,None,["Category_1_Illegal_Activity", "Category_3_Hate_Harass_Violence", "Category_4_Malware", "Category_5_Physical_Harm", "Category_6_Economic_Harm", "Category_7_Fraud_Deception", "Category_8_Adult_Content", "Category_9_Political_Campaigning", "Category_10_Privacy_Violation_Activity", "Category_11_Tailored_Financial_Advice"])
        configs['attaq']=('ibm-research/AttaQ',None,"input",'train')
        configs['openai'] = ("mmathys/openai-moderation-api-evaluation", None, "prompt", 'train')
    pprint(configs)

    # Process each dataset (Each dataset configuration generates an independent save file containing only the processing results of the corresponding dataset, with complete separation between files)
    for name, (src, subset, key, split_type) in configs.items():
        ### 'key' refers to the prompt_key in the dataset
        label_fn = get_label_fn(src, subset) # Get the corresponding label function

        # Process the dataset
        df = process_prompt_dataset(src, subset, key, label_fn, tokenizer, max_length, name, split=split_type, args = args)

        # Allow sampling with replacement for toxic datasets (to ensure sufficient quantity), disallow for other datasets
        allow_replace = True if name == 'toxic' else False

        # Do not balance test sets; balance training sets
        if split == 'test':
            #df_balanced = df
            df_balanced = balance_dataset(df, num_safe=num_safe, num_unsafe=num_unsafe, allow_replace=allow_replace)
        else:
            df_balanced = balance_dataset(df, num_safe=num_safe, num_unsafe=num_unsafe, allow_replace=allow_replace)

        # Format the dataset
        formatted = format_prompt_dataset(df_balanced, name, split, data_source)

        # Save in parquet format (efficient columnar storage format)
        path = os.path.join(local_dir, f"{name}_prompt_{split}.parquet")
        formatted.to_parquet(path)

        # Calculate and print save information
        safe_count = (df_balanced['answer'] == 'safe').sum()
        unsafe_count = (df_balanced['answer'] == 'unsafe').sum()
        print(f"✅ Saved {len(formatted)} samples to {path} (safe: {safe_count}, unsafe: {unsafe_count})")

def main():
    """Main function: Parse command-line arguments and start the dataset preparation process"""
    parser = argparse.ArgumentParser() # Command-line argument parser
    parser.add_argument('--target_hdfs_path_dir', type=str, default=None) # Used to specify the target HDFS save path, but not actually used in the current code (HDFS is a distributed file system commonly used for large-scale data storage and processing, especially in cluster environments)
    parser.add_argument('--local_dir', type=str, default='datasets_rsafe',help="local save direction")
    parser.add_argument('--split', type=str, default='train',help="split of the dataset to process, e.g., 'train', 'test'")
    #parser.add_argument('--split', type=str, default='test',help="split of the dataset to process, e.g., 'train', 'test'")
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--num_safe', type=int, default=3000, help='Number of safe samples')
    parser.add_argument('--num_unsafe', type=int, default=3000, help='Number of unsafe samples')
    # parser.add_argument('--num_safe', type=int, default=1500, help='Number of safe samples')
    # parser.add_argument('--num_unsafe', type=int, default=1500, help='Number of unsafe samples')
    parser.add_argument('--HF_token', type=str, default='xxxxxx',help="HF token to access the dataset") 
    ### data_source
    parser.add_argument('--data_source', type=str, default='checklist_reward',help="HF token to access the dataset")

    args = parser.parse_args()

    assert args.data_source in ['checklist_reward', 'rsafe'], "data_source can only in ['checklist_reward', 'rsafe']"
    login(token=args.HF_token)
    prepare_all_prompt_datasets(
        args.local_dir,
        args.max_length,
        args.num_safe,
        args.num_unsafe,
        args.split,
        args.data_source,
        args
    )

if __name__ == '__main__':

    main()