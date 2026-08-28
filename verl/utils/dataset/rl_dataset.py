# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import ListConfig
import os
from typing import List, Union, Optional, Callable
import copy
import datasets
from collections import defaultdict
import logging
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

# Configure logging
LOG_FILE = '~/verl/examples/custom_task/data.log'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Get the logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create and add file handler
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Create and add console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Test logging
logger.info("=== RLHFDataset module initialized ===")

def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key: str = 'prompt',
                 image_key: str = 'images',
                 max_prompt_length: int = 1024,
                 cache_dir: str = '~/.cache/verl/rlhf',
                 chat_template_func: Optional[Callable] = None,
                 return_raw_chat: bool = False,
                 truncation: str = 'right',
                 filter_overlong_prompts: bool = False,
                 num_workers: Optional[int] = None):
        
        # Test log message
        logger.info("=== Starting new RLHFDataset initialization ===")
        
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        logger.info("\n[INIT] Initializing RLHFDataset with:")
        logger.info(f"- Parquet files: {parquet_files}")
        logger.info(f"- Prompt key: {prompt_key}")
        logger.info(f"- Return raw chat: {return_raw_chat}")
        logger.info(f"- Max prompt length: {max_prompt_length}")
        logger.info(f"- Truncation mode: {truncation}")
        
        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts
        if num_workers is None:
            self.num_workers = max(1, os.cpu_count() // 4)
        else:
            self.num_workers = min(num_workers, os.cpu_count())

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            logger.info(f"\n[READ] Reading parquet file: {parquet_file}")
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        logger.info(f'\n[DATA] Dataset length: {len(self.dataframe)}')
        logger.info(f'[DATA] Dataset columns: {list(self.dataframe.features.keys())}')
        
        # Print an example of the input data
        if len(self.dataframe) > 0:
            logger.info("\n[DATA] First row example:")
            example = self.dataframe[0]
            for key, value in example.items():
                if isinstance(value, (str, int, float, bool)):
                    logger.info(f"- {key}: {value}")
                else:
                    logger.info(f"- {key}: {type(value)}")

        # Instead of filtering, we'll use truncation in __getitem__
        if self.filter_overlong_prompts:
            logger.info("\n[NOTICE] filter_overlong_prompts is set to True but truncation is enabled.")
            logger.info(f"Long prompts will be truncated to {self.max_prompt_length} tokens instead of being filtered.")

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            logger.info(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        chat = row_dict.pop(self.prompt_key)
        
        # if item == 0:  # Only log for the first item to avoid spam
        #     logger.info("\n[PROCESS] Processing item 0:")
        #     logger.info(f"- Raw chat from prompt_key: {chat}")
        #     logger.info(f"- Type of chat: {type(chat)}")

        # Format chat as a list of message dictionaries if it's a string
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
            logger.info(f"- Formatted chat as message dict: {chat}")
        
        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        
        # if item == 0:
        #     logger.info(f"- After applying chat template: {prompt_with_chat_template}")
        #     logger.info(f"- Type of template result: {type(prompt_with_chat_template)}")
        #     logger.info(f"- Tokenizer chat template: {self.tokenizer.chat_template}")

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)
        
        
        if item == 0:
            logger.info(f"- Tokenized input_ids shape: {input_ids.shape}")
            logger.info(f"- Decoded tokens: {self.tokenizer.decode(input_ids[0])}")

        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            logger.info("\n[RAW_PROMPT] Setting raw_prompt in row_dict")
            logger.info(f"- Original chat: {chat}")
            row_dict['raw_prompt'] = chat
            logger.info(f"- Successfully set raw_prompt in row_dict")

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
