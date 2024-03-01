from __future__ import annotations

import re
import torch
import torch.nn as nn
from typing import Literal
import numpy as np

from .base import VideoRewardBase
from .clip import CLIP
from .pooling import TemporalPooling
from .head import CLIPScoreHead


class MineCLIP(VideoRewardBase):
    def __init__(
        self,
        arch: str,  # vit_base_p16
        *,
        resolution: tuple[int, int],
        pool_type: str,
        image_feature_dim: int,
        mlp_adapter_spec: str,
        hidden_dim: int,
    ):
        """
        Args:
          mlp_adapter_spec: v3-1.t2 means on the vision branch, 3 MLP layers of image
            adapter (before video pooling), 1 layer of video adapter (after pooling).
            On text branch, 2 layers of text adapter
        """
        self.arch = arch
        VIDEO_SEQ_LEN = 32
        assert arch.startswith("vit_base_p16")
        assert image_feature_dim == 512
        clip_config = {
            "context_length": 77,
            "embed_dim": 512,
            "image_resolution": 224,    # Change to 160*256
            "text_heads": 8,
            "text_layers": 12,
            "text_width": 512,
            "vision_layers": 12,
            "vision_patch_size": 16,
            "vision_width": 768,
            "vocab_size": 49408,
        }
        model = CLIP(**clip_config)
        model.vision_model.resize_pos_embed(resolution)

        # regex match v3-1.t2
        m = re.match(
            r"v(?P<image_adapter>\d+)"
            r"-(?P<video_adapter>\d+)"
            r"\.t(?P<text_adapter>\d+)",
            mlp_adapter_spec,
        )
        image_adapter_layers, video_adapter_layers, text_adapter_layers = (
            int(m.group("image_adapter")),
            int(m.group("video_adapter")),
            int(m.group("text_adapter")),
        )

        assert image_feature_dim == hidden_dim

        temporal_encoder = TemporalPooling(
            pool_type=pool_type,
            input_dim=image_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            layers_before_pool=image_adapter_layers,
            max_seq_len=VIDEO_SEQ_LEN,
        )
        if not isinstance(temporal_encoder.mlp_before_pool, nn.Identity):
            for module in temporal_encoder.mlp_before_pool:
                # initialize linear layers as identity matrices
                if isinstance(module, nn.Linear):
                    module.weight.data.copy_(torch.eye(module.weight.shape[0]))
                    module.bias.data.zero_()
        reward_head = CLIPScoreHead(
            model,
            video_adapter_layers=video_adapter_layers,
            text_adapter_layers=text_adapter_layers,
            feature_dim=image_feature_dim,
        )

        super().__init__(
            image_encoder=model.vision_model,
            temporal_encoder=temporal_encoder,
            reward_head=reward_head,
        )
        self.clip_model = model
        
        self.temporal_encoder = temporal_encoder
        self.reward_head = reward_head
        self.video_adapter_layers = video_adapter_layers
        self.text_adapter_layers = text_adapter_layers
        self.video_layers = self.clip_model.vision_model.layers + self.temporal_encoder.layers + self.video_adapter_layers   # (12+2) + 1 + 2
        self.text_layers = self.clip_model.text_model.layers + self.text_adapter_layers  # (12+2) + 0
        self.cross_layers = 1
        self.layers = self.cross_layers + max(self.video_layers, self.text_layers)

    def get_layer(self, layer: int, layer_type: Literal['video', 'text', 'cross']):
        print("----------------Getting layer------------- ", self.video_layers, self.text_layers, self.cross_layers)
        if layer_type == 'video':
            if layer < self.clip_model.vision_model.layers:
                print("Layer index video: ", layer)
                return self.clip_model.vision_model.get_layer(layer)

            elif layer < self.clip_model.vision_model.layers + self.temporal_encoder.layers:
                print("Layer index video: ", layer)
                return self.temporal_encoder.get_layer(layer - self.clip_model.vision_model.layers)
            elif layer < self.video_layers: # 1st
                print("Layer index video: ", layer)
                return self.reward_head.get_layer(layer - self.clip_model.vision_model.layers - self.temporal_encoder.layers, type='video')
            # - self.video_adapter_layers
        elif layer_type == 'text':
            if layer < self.clip_model.text_model.layers:
                print("Layer index text: ", layer)              
                return self.clip_model.text_model.get_layer(layer)
            elif layer < self.text_layers:
                print("Layer index text: ", layer) 
                return self.reward_head.get_layer(layer - self.clip_model.text_model.layers, type='text')
        elif layer_type == 'cross':
            if layer == 0:
                print("Layer index cross: ", layer) 
                return nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                # return nn.Parameter(self.clip_model.logit_scale.cpu()) 
                # self.clip_model.logit_scale  # TODO: check this 2 versions: build_logit_scale
            print("Layer total cross: done", layer)
        return []
    
    def encode_text(self, text_tokens):
        return self.clip_model.encode_text(text_tokens)

    def encode_video(self, videos):
        return self.forward_video_features(self.forward_image_features(videos))

    def clamp_logit_scale(self):
        """
        Clamp the logit scale to avoid numerical instability
        """
        self.clip_model.clamp_logit_scale()
