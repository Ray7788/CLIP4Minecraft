from typing import Literal
import torch
import torch.nn as nn

from mineclip.utils import build_mlp


class CLIPScoreHead(nn.Module):
    """
    Head for computing reward scores from video and text features
    Args:
      clip_model: CLIP model
      video_adapter_layers: number of layers for video adapter
      text_adapter_layers: number of layers for text adapter
      feature_dim: feature dimension
    """
    def __init__(
        self,
        clip_model,
        *,
        video_adapter_layers,
        text_adapter_layers,
        feature_dim,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.video_residual_weight = None
        self.text_residual_weight = None

        # video adapter
        if video_adapter_layers == 0:   # if no adapter layers, use identity function
            self.video_adapter = nn.Identity()
        else:
            self.video_adapter = build_mlp(
                # dim of input, output, hidden layers are all feature_dim, which is 512
                input_dim=feature_dim,
                output_dim=feature_dim,
                hidden_dim=feature_dim,
                num_layers=video_adapter_layers,
                add_input_activation=False,
            )
            self.video_residual_weight = nn.Parameter(torch.tensor(4.0))    # initialize res_weight to be positive so sigmoid(res_weight) is close to 1
        # text adapter
        if text_adapter_layers == 0:
            self.text_adapter = nn.Identity()
        else:
            self.text_adapter = build_mlp(
                input_dim=feature_dim,
                output_dim=feature_dim,
                hidden_dim=feature_dim,
                num_layers=text_adapter_layers,
                add_input_activation=False,
            )
            # input * sigmoid(res_weight) + MLP(input) * (1-sigmoid(res_weight))
            # initialize res_weight to be positive so sigmoid(res_weight) is close to 1
            self.text_residual_weight = nn.Parameter(torch.tensor(4.0))
        self.layers = 1 + 1 # video and text adapter layers?

    def get_layer(self, layer: int, type: Literal['video', 'text']):
        # assert layer == 0, "layer must be 0, but got {},\n video adapter layer are {},\n text adapter layer are {}. ".format(layer, self.video_adapter, self.text_adapter)
        if type == 'video':
            assert layer < len(self.video_adapter), "Invalid layer index for video adapter"
            return self.video_adapter[layer], self.video_residual_weight
        elif type == 'text':
            assert layer == 0, "Invalid layer index for text adapter"        
            return self.text_adapter, self.text_residual_weight
        else:   # return self.adapter, self.residual_weight
            return self.video_adapter, self.video_residual_weight, self.text_adapter, self.text_residual_weight

    def forward(self, video_feature, texts):
        if self.video_residual_weight is None:
            adapted_img = self.video_adapter(video_feature)
        else:
            # input * sigmoid(res_weight) + MLP(input) * (1-sigmoid(res_weight))
            res = torch.sigmoid(self.video_residual_weight)
            adapted_img = res * video_feature + (1.0 - res) * self.video_adapter(
                video_feature
            )
        
        text_feature = self.clip_model.encode_text(texts)
        if self.text_residual_weight is None:
            adapted_text = self.text_adapter(text_feature)
        else:
            # input * sigmoid(res_weight) + MLP(input) * (1-sigmoid(res_weight))
            # similar to residual connection
            res = torch.sigmoid(self.text_residual_weight)
            adapted_text = res * text_feature + (1.0 - res) * self.text_adapter(
                text_feature
            )
        # compute reward scores
        logits_per_video, logits_per_text = self.clip_model(adapted_img, adapted_text)
        return logits_per_video, logits_per_text
