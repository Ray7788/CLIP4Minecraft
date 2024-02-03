import cv2
import numpy as np
import torch

def video_to_tensor(video_paths, num_frames=16, height=160, width=256, batch_size=6):
    """
    Args:
        video_paths: list of paths to video files
        num_frames: number of frames to sample from each video
        height: height of the frames(resolution 160*256)
        width: width of the frames
        batch_size: batch size
    Returns:
        video_tensor: tensor of shape (batch_size, num_frames, num_channels, height, width)
    """
    batch = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)

        assert cap.isOpened(), f"Failed to open video: {video_path}"
        frames = [] # list to store frames, in total 16 frames

        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Change color space from BGR to RGB
            # Change resolution to 160*256
            frame = cv2.resize(frame, (width, height))
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            frames.append(frame)
        cap.release()

        if len(frames) < num_frames:
            # If video has less frames than required, duplicate the last frame
            # while len(frames) < num_frames:
            #     frames.append(frames[-1])

            # If video is shorter than sequence_length, repeat frames
            frames = frames * (num_frames // len(frames)) + frames[:num_frames % len(frames)]
  
        # Stack 16 frames on the first dimension
        video_tensor = torch.stack(frames)
        # video_tensor = torch.stack([torch.from_numpy(frame) for frame in frames])

        # Rearrange dimensions to: (batch_size, num_channels, height, width)
        video_tensor = video_tensor.permute(0, 3, 1, 2)        
        batch.append(video_tensor)

        # if batch is full(i.e. batch size is 6), yield it and reset batch
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch = []
    
    # Yield remaining frames if the last batch is smaller than batch_size
    if batch:
        yield torch.stack(batch)

    # return video_tensor

# video 是一个 PyTorch 张量，其形状为 (6, 16, 3, 160, 256)。这表示有 6 个视频，每个视频有 16 帧，每帧是一个 160x256 的 RGB 图像（3 个颜色通道）。视频的像素值在 0 到 255 之间。