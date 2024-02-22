import torch
from torch import nn


class MHCA(nn.Module):
    """
    Multi-Head Channel Attention
    https://arxiv.org/pdf/2207.05501.pdf
    """

    def __init__(self, in_features: int, head_dim: int = 32):
        super().__init__()
        n_heads = in_features // head_dim
        self.mhca = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, padding=1, groups=n_heads),
            nn.BatchNorm2d(num_features=in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mhca(x)
