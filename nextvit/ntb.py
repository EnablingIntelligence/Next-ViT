import torch
from torch import nn

from nextvit.attention import MHCA, EMHSA
from nextvit.mlp import MLP


class NTB(nn.Module):
    """
    Next Transformer Block
    https://arxiv.org/pdf/2207.05501.pdf
    """

    def __init__(self, in_features: int, out_features: int, spatial_reduction_ratio: int = 1,
                 shrink_ratio: float = 0.75, mlp_expansion_ratio: int = 2):
        super().__init__()
        emhsa_features = int(out_features * shrink_ratio)
        mhca_features = out_features - emhsa_features

        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=emhsa_features, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=emhsa_features, out_channels=mhca_features, kernel_size=1)
        self.emhsa = EMHSA(emhsa_features, spatial_reduction_ratio)
        self.mhca = MHCA(mhca_features)
        self.mlp = MLP(out_features, out_features, mlp_expansion_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emhsa_in = self.conv1(x)
        emhsa_out = self.emhsa(emhsa_in) + emhsa_in
        mhca_in = self.conv2(emhsa_out)
        mhca_out = self.mhca(mhca_in) + mhca_in
        mlp_in = torch.cat([emhsa_out, mhca_out], dim=1)
        return self.mlp(mlp_in) + mlp_in
