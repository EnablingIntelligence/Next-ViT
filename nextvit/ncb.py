import torch
from torch import nn

from nextvit.attention import MHCA
from nextvit.mlp import MLP


class NCB(nn.Module):
    """
    Next Convolution Block
    https://arxiv.org/pdf/2207.05501.pdf
    """

    def __init__(self, in_features: int, out_features: int, expansion_ratio: int = 1):
        super().__init__()
        self.mhca = MHCA(in_features)
        self.mlp = MLP(in_features, out_features, expansion_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.mhca(x) + x
        return self.mlp(features) + features
