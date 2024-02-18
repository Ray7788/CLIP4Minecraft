import torch
from torch.utils.data import Dataset
from typing import Literal
from .data_utils import get_processed_len, load_processed_data

class NaiveDataset(Dataset):
    """
    Build a naive dataset of processed data.

    Args:
        dataset: which category of dataset to use, train or test.
        dataset_log_file: path to the corresponding dataset file.
        use_mask: whether to use entity mask and action mask of text input.
    """
    def __init__(self, dataset_log_file: str, use_mask: bool = True, dataset: Literal["train", "test"] = None):
        self.dataset = dataset
        self.dataset_log_file = dataset_log_file
        self.len = get_processed_len(dataset_log_file, dataset)
        self.use_mask = use_mask

    def __len__(self):
        """
        Returns:
            int: length of the dataset: number of video-text pairs.
        """
        return self.len

    def __getitem__(self, idx):
        """
        Get the idx-th video-text pair.
        
        Args:
            idx: index of the video-text pair.

        Returns:
            video: torch.Tensor, video input.
            text: string, text input.
        """
        data = list(load_processed_data(self.dataset_log_file, idx, self.use_mask, self.dataset))
        video = data[0] 
        text = data[1]

        return video, text
