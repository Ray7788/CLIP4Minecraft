from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from functools import lru_cache
from typing import *
import numpy as np
import torch
import yaml
import json


# ------------------------ Data Loader Helper Functions ------------------------ #
@lru_cache(None)
def get_processed_list(dataset_log_file: str, dataset: Literal['train', 'test'] = None):
    """
    Get the JSON of processed data files.
    """
    with open(dataset_log_file, 'r') as f:
        datasets = json.load(f)

    return datasets


def get_processed_len(dataset_log_file, dataset: Literal["train", "test"] = None):
    """
    Get the length of processed data files.
    """
    return len(get_processed_list(dataset_log_file, dataset))


def load_processed_data(dataset_log_file, data_idx: int, use_mask: bool = False,
                        dataset: Literal["train", "test"] = None):
    """
    Load processed data(video encoding and raw text) based on the data_idx of the JSON file.

    Args:
        dataset_log_file: The JSON file that contains the processed data files.
        data_idx: The index of the data to be loaded.
        use_mask: Whether to use the mask in the text input.
        dataset: The dataset to be loaded. If None, all datasets will be loaded.

    Returns:
        video_input: The video input: torch.Tensor
        text_input: The text input: raw text
    """
    processed_list = get_processed_list(dataset_log_file, dataset)

    assert data_idx < get_processed_len(processed_list), \
        f'Index {data_idx} is beyond the length of processed {dataset} dataset, {len(processed_list)}.'

    # specified video id
    file_path = processed_list[data_idx].get('vid')

    text_input = processed_list[data_idx].get('transcript clip')
    # with open(os.path.join(file_path, 'text_input.pkl'), 'rb') as f:
    #     text_input = torch.load(f)

    with open(os.path.join(file_path, '.pth'), 'rb') as f:
        video_input = torch.load(f)
    
    return (video_input, *text_input) if use_mask else (video_input, text_input)
