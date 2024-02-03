import os
from typing import Literal
import cv2
import torch
from torchvision import transforms
import numpy as np
import warnings
import yaml
from PIL import Image

from __future__ import annotations
from mineclip.utils.image_utils import basic_image_tensor_preprocess
from mineclip.utils.convert_utils import any_to_torch_tensor


def img_to_tensor(frame, dtype=None, device=None):
    """
    Convert a numpy array to a torch tensor with shape (1, C, H, W).
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Change color space from BGR to RGB
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(frame.tobytes()))
    img = img.view(frame.shape[1], frame.shape[0], len(frame.mode)) # HWC ordinary format
    img = img.permute((2, 0, 1)).contiguous()   # change to CHW (PyTorch format)
    img = any_to_torch_tensor(img, dtype=dtype, device=device)
    img.unsqueeze_(dim=0)   # add a leading batch dim

    return img


# 辅助函数，将numpy数组转换为torch张量
# def any_to_torch_tensor(data, dtype=None, device=None):
#     if not torch.is_tensor(data):
#         data = torch.tensor(data, dtype=dtype, device=device)
#     return data


def sample_frames(video_file, num_frames=16, frame_size=(160, 256)):
    """
    1. From a given video file, sample 16 frames uniformly across the video and return them as a list of numpy arrays.
    """
    cap = cv2.VideoCapture(video_file)
     
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_file}")
 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
    frame_indices = [int(round(i * total_frames / num_frames)) for i in range(num_frames)]

    sampled_frames = []
    try:
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Could not read frame at index {frame_index} from video file {video_file}")
            
            # Resize the frame to the specified size
            frame = cv2.resize(frame, frame_size)

            sampled_frames.append(frame)
    finally:
        cap.release()
    
    sampled_frames = np.array(sampled_frames)

    return sampled_frames


# for training only: data augmentation
image_transform = transforms.Compose([
    transforms.RandomResizedCrop((160, 256), scale=(0.2, 1.), interpolation=transforms.InterpolationMode.BICUBIC),
])


def preprocess_frames(frames, dataset: Literal["train", "test"] = None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), shape=(160, 256)):
    """
    2. Preprocess a list of frames, including resizing, data augmentation, and normalization.
    """
    preprocessed_frames = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if dataset == "train":  # Data augmentation for training
            image_transform(frame)

        frame_tensor = img_to_tensor(frame)
        preprocessed_frames.append(frame_tensor)
        
    return torch.stack(preprocessed_frames) # Stack the frames into a single tensor


def preprocess_video(video_file, num_frames=16, dataset: Literal["train", "test"] = None):
    """
    Preprocess a video file by sampling 16 frames and preprocessing them
    """
    frames = sample_frames(video_file, num_frames=num_frames)
    preprocessed_frames = preprocess_frames(frames, dataset)

    pth_id = os.path.splitext(video_file)[0]
    torch.save(preprocessed_frames, pth_id + ".pth")

"""
Logic: 
MP4 video -> sample 16 frames -> preprocess each frame and convert to tensor
-> stack 16 frames -> preprocess_video: save as .pth file
"""