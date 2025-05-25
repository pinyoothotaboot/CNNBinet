import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryLinear(nn.Linear):
    """
    Binary Linear Layer (Hybrid)

    This layer applies a linear transformation to the incoming data
    using binarized weights scaled by their mean absolute value (per output neuron).

    Forward: y = Linear(input, sign(W) * alpha + bias)

    - Weights are binarized to {-1, +1}
    - Alpha is a scaling factor: mean(abs(W), dim=1)
    - Keeps bias in full precision
    """

    def forward(self, input):
        alpha = torch.mean(torch.abs(self.weight), dim=1, keepdim=True)
        bin_weight = self.weight.sign() * alpha
        return F.linear(input, bin_weight, self.bias)


class BinaryConv2d(nn.Conv2d):
    """
    Binary Convolution Layer (Hybrid)

    Performs a 2D convolution with binarized weights scaled by
    mean absolute value (per output channel).

    Forward: y = Conv2D(input, sign(W) * alpha + bias)

    - Weights are binarized to {-1, +1}
    - Alpha is calculated per output channel: mean(abs(W), dim=(1,2,3))
    - Keeps bias in full precision
    """

    def forward(self, input):
        alpha = torch.mean(torch.abs(self.weight), dim=(1, 2, 3), keepdim=True)
        bin_weight = self.weight.sign() * alpha
        return F.conv2d(
            input, bin_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
