#!/usr/bin/env python
# Combined script to generate response classification datasets

import os
import pandas as pd
import argparse
import json
from datasets import load_dataset, concatenate_datasets
from tqdm.auto import tqdm # For displaying progress bar
from transformers import AutoTokenizer
from label_response_fn import get_label_fn # Custom function to get data labels
import response_templates 
from huggingface_hub import login
from pprint import pprint

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


def process_prompt_dataset(dataset_name, subset, prompt_key, response_key, label_fn, tokenizer, max_length, dataset_tag,
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
        dataset_dict = load_dataset(dataset_name, subset) if subset else load_dataset(dataset_name)
        all_splits_with_safety = []
        for split_name in split:
            if split_name not in dataset_dict:
                raise KeyError(f"Split '{split_name}' does not exist! Available splits: {list(dataset_dict.keys())}")
            split_dataset = dataset_dict[split_name]
            if "safe" in split_name:
                safety_label = "safe"
            elif "unsafe" in split_name:
                safety_label = "unsafe"
            else:
                safety_label = "unknown"  # Or raise ValueError(f"Split name {split_name} has no matching safety label")
            # Add 'safety' column to all samples in the current split
            split_dataset = split_dataset.add_column(
                "safety", 
                [safety_label] * len(split_dataset)  # Fill safety label for each sample
            )
            all_splits_with_safety.append(split_dataset)
        dataset = concatenate_datasets(all_splits_with_safety)
    else:
        dataset = load_dataset(dataset_name, subset)[split] if subset else load_dataset(dataset_name)[split]

    data, skipped = [], 0 # 'data' stores qualified processed data; 'skipped' counts samples skipped due to excessive length

    # Iterate through the dataset with progress bar
    for item in tqdm(dataset, desc=f"Processing {dataset_tag}"):
        if dataset_name=='fedric95/T2TSyntheticSafetyBench':
            if item['class']=='safe_unreleated_unsafe':
                continue

        # Extract prompt text
        prompt = item[prompt_key]
        response = item[response_key]

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
        if dataset_name in ["PKU-Alignment/PKU-SafeRLHF", "PKU-Alignment/BeaverTails"]:
            category_list = category.split('@@@') # Split category into a list by comma
            if len(category_list) > 1:
                continue

        # Format input prompt and category dictionary using templates
        chat_prompt, category_dict_rt = response_templates.format_input_prompt(prompt, response, label, dataset_name, subset, dataset_tag, split = None, sample_ratio = sample_ratio, args = args)
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
    
def format_prompt_dataset(df, dataset_tag, data_source, split):
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
                ### Convert to JSON format to avoid pyarrow errors (schema mismatch if it is a dict across multiple train data files), save for more advanced reward usage
                "category_dict": json.dumps(category_dict) # Convert category dictionary to JSON string to avoid formatting issues in subsequent processing
            }
        })
    return pd.DataFrame(output)

def prepare_all_prompt_datasets(local_dir, max_length, num_safe, num_unsafe, split='train', args = None):
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
    # Dataset configuration: Name -> (Data source, Subset, Prompt key, response key, Split)
    configs = {
        'PKU-SafeRLHF_default_0': ("PKU-Alignment/PKU-SafeRLHF",'default','prompt','response_0',split),
        'PKU-SafeRLHF_default_1': ("PKU-Alignment/PKU-SafeRLHF",'default','prompt','response_1',split),
    }
    # Add different datasets based on split type
    if split == 'train':
        configs['BeaverTails'] = ('PKU-Alignment/BeaverTails', None, "prompt", 'response', '330k_train')

    elif split == 'test':
        configs['BeaverTails'] = ('PKU-Alignment/BeaverTails', None, "prompt", 'response', '30k_test')
        configs['T2T'] = ('fedric95/T2TSyntheticSafetyBench',None, "question", "answer", 'train')
        configs['do_not_answer'] = ('LibrAI/do-not-answer', None, "question", "ChatGLM2_response", 'train')

    pprint(configs)

    # Process each dataset (Each dataset configuration generates an independent save file containing only the processing results of the corresponding dataset, with complete separation between files)
    for name, (src, subset, key, response_key, split_type) in configs.items():
        ### 'key' refers to the prompt_key in the dataset
        label_fn = get_label_fn(name, src, subset) # Get the corresponding label function

        # Process the dataset
        df = process_prompt_dataset(src, subset, key, response_key, label_fn, tokenizer, max_length, name, split=split_type, args = args)

        # Allow sampling with replacement for toxic datasets (to ensure sufficient quantity), disallow for other datasets
        allow_replace = True if name == 'toxic' else False

        # Do not balance test sets; balance training sets
        if split == 'test':
            #df_balanced = df
            df_balanced = balance_dataset(df, num_safe=num_safe, num_unsafe=num_unsafe, allow_replace=allow_replace)
        else:
            df_balanced = balance_dataset(df, num_safe=num_safe, num_unsafe=num_unsafe, allow_replace=allow_replace)

        # Format the dataset
        formatted = format_prompt_dataset(df_balanced, name, args.data_source, split)

        # Save in parquet format (efficient columnar storage format)
        path = os.path.join('..',local_dir, f"{name}_prompt_{split}.parquet")
        print(path)
        #path = os.path.join(local_dir, f"{name}_prompt_{split}.parquet")
        formatted.to_parquet(path)

        # Calculate and print save information
        safe_count = (df_balanced['answer'] == 'safe').sum()
        unsafe_count = (df_balanced['answer'] == 'unsafe').sum()
        print(f"✅ Saved {len(formatted)} samples to {path} (safe: {safe_count}, unsafe: {unsafe_count})")

def main():
    """Main function: Parse command-line arguments and start the dataset preparation process"""
    parser = argparse.ArgumentParser() # Command-line argument parser
    parser.add_argument('--target_hdfs_path_dir', type=str, default=None) 
    parser.add_argument('--local_dir', type=str, default='datasets_rsafe',help="local save direction")
    parser.add_argument('--split', type=str, default='train',help="split of the dataset to process, e.g., 'train', 'test'")
    #parser.add_argument('--split', type=str, default='test',help="split of the dataset to process, e.g., 'train', 'test'")
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--num_safe', type=int, default=3000, help='Number of safe samples')
    parser.add_argument('--num_unsafe', type=int, default=3000, help='Number of unsafe samples')
    # parser.add_argument('--num_safe', type=int, default=1500, help='Number of safe samples')
    # parser.add_argument('--num_unsafe', type=int, default=1500, help='Number of unsafe samples')
    #parser.add_argument('--wo_category', action='store_true', help='Use template without safety categories', default=False)
    parser.add_argument('--HF_token', type=str, default='xxxxxxxxx',help="HF token to access the dataset") 
    ### data_source
    parser.add_argument('--data_source', type=str, default='rsafe',help="HF token to access the dataset")

    args = parser.parse_args()

    assert args.data_source in ['checklist_reward', 'rsafe'], "data_source can only in ['checklist_reward', 'rsafe']"

    login(token=args.HF_token)
    prepare_all_prompt_datasets(
        args.local_dir,
        args.max_length,
        args.num_safe,
        args.num_unsafe,
        args.split,
        args
    )

if __name__ == '__main__':

    main()