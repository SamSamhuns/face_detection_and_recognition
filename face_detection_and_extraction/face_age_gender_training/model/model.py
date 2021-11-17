import torch
import torch.nn as nn
from base import BaseModel


class Residual(nn.Module):
    """
    This module looks like what you find in the original resnet or IC paper
    (https://arxiv.org/pdf/1905.05928.pdf), except that it's based on MLP, not CNN.
    If you flag `only_MLP` as True, then it won't use any batch norm, dropout, or
    residual connections
    """

    def __init__(self, num_features: int, dropout: float,
                 add_residual: bool, add_IC: bool, i: int, j: int):
        super().__init__()
        self.num_features = num_features
        self.add_residual = add_residual
        self.add_IC = add_IC
        self.i = i
        self.j = j

        if (not ((self.i == 0) and (self.j == 0))) and self.add_IC:
            self.norm_layer1 = nn.BatchNorm1d(num_features)
            self.dropout1 = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(num_features, num_features)
        self.relu1 = nn.ReLU()

        if self.add_IC:
            self.norm_layer2 = nn.BatchNorm1d(num_features)
            self.dropout2 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(num_features, num_features)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.Tensor:

        identity = out = x

        if (not ((self.i == 0) and (self.j == 0))) and self.add_IC:
            out = self.norm_layer1(out)
            out = self.dropout1(out)
        out = self.linear1(out)
        out = self.relu1(out)

        if self.add_IC:
            out = self.norm_layer2(out)
            out = self.dropout2(out)
        out = self.linear2(out)

        if self.add_residual:
            out += identity

        out = self.relu2(out)
        return out


class DownSample(nn.Module):
    """
    This module is an MLP, where the number of output features is lower than
    that of input features. If you flag `only_MLP` as False, it'll add norm
    and dropout
    """

    def __init__(self, in_features: int, out_features: int, dropout: float,
                 add_IC: bool):
        super().__init__()
        assert in_features > out_features

        self.in_features = in_features
        self.out_features = out_features
        self.add_IC = add_IC

        if self.add_IC:
            self.norm_layer = nn.BatchNorm1d(in_features)
            self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.Tensor:
        out = x

        if self.add_IC:
            out = self.norm_layer(out)
            out = self.dropout(out)
        out = self.linear(out)
        out = self.relu(out)
        return out


class ResMLP(BaseModel):
    """
    MLP with optinally batch norm, dropout, and residual connections. I got
    inspiration from the original ResNet paper and https://arxiv.org/pdf/1905.05928.pdf.

    Downsampling is done after every block so that the features can be encoded
    and compressed.
    """

    def __init__(self, dropout: float, num_residuals_per_block: int, num_blocks: int, num_classes: int,
                 num_initial_features: int, reduce_in_features: int, add_residual: bool = True, add_IC: bool = True):
        super().__init__()

        blocks = []
        # input feature space reduction layer, acts as encoder layer
        # if reduce_feat_num is not None, reduce input features with downsampling instead of residual block
        if reduce_in_features is not None:
            blocks.append(DownSample(
                num_initial_features, reduce_in_features, dropout, add_IC))
        else:
            reduce_in_features = num_initial_features

        for i in range(num_blocks):
            blocks.extend(self._create_block(
                reduce_in_features, dropout, num_residuals_per_block, add_residual, add_IC, i))
            reduce_in_features //= 2

        # last classification layer
        blocks.append(nn.Linear(reduce_in_features, num_classes))
        self.blocks = nn.Sequential(*blocks)

    def _create_block(self, in_features: int, dropout: float,
                      num_residuals_per_block: int, add_residual: bool,
                      add_IC: bool, i: int) -> list:
        block = []
        for j in range(num_residuals_per_block):
            block.append(Residual(in_features, dropout,
                                  add_residual, add_IC, i, j))
        block.append(DownSample(
            in_features, in_features // 2, dropout, add_IC))
        return block

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return self.blocks(x)


class FullyConnectedNet(BaseModel):
    """
    Classic fully connected neural network that downsamples features by half every layer
    """

    def __init__(self, num_blocks: int, num_classes: int,
                 num_initial_features: int, reduce_in_features: int, **kwargs):
        super().__init__()

        blocks = []
        # input feature space reduction layer, acts as encoder layer
        # if reduce_feat_num is not None, reduce input features with downsampling instead of residual block
        if reduce_in_features is not None:
            blocks.append(nn.Linear(num_initial_features, reduce_in_features))
            blocks.append(nn.ReLU())
        else:
            reduce_in_features = num_initial_features

        for i in range(num_blocks):
            blocks.extend(self._create_block(reduce_in_features))
            reduce_in_features //= 2

        # last classification layer
        blocks.append(nn.Linear(reduce_in_features, num_classes))
        self.blocks = nn.Sequential(*blocks)

    def _create_block(self, in_features: int) -> list:
        block = []
        block.append(nn.Linear(in_features, in_features // 2))
        block.append(nn.ReLU())
        return block

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return self.blocks(x)
