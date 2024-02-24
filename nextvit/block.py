import torch
from torch import nn

from nextvit.ncb import NCB
from nextvit.ntb import NTB


class NextViTBlock(nn.Module):
    """
    Next Vision Transformer Block
    https://arxiv.org/pdf/2207.05501.pdf
    """

    def __init__(self, in_features: int, out_features: int, num_ncb_layers: int, num_ntb_layers: int, depth: int = 1,
                 sr_ratio: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_ncb_layers = num_ncb_layers
        self.num_ntb_layers = num_ntb_layers
        self.depth = depth
        self.spacial_reduction_ratio = sr_ratio

        self.block = self.__make_block_layers()

    def __make_block_layers(self) -> nn.Sequential:
        block = []

        for depth_idx in range(self.depth):
            is_last_layer = depth_idx == self.depth - 1
            out_features = self.out_features if is_last_layer else self.in_features
            block.append(self.__make_layer(out_features))

        return nn.Sequential(*block)

    def __make_layer(self, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            *[
                *[NCB(self.in_features, self.in_features) for _ in range(self.num_ncb_layers)],
                *[NTB(self.in_features, out_features, self.spacial_reduction_ratio) for _ in range(self.num_ntb_layers)]
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
