from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer


class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride, dropout=dropout, conv_only=True
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetResBlockFused(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
            bias=norm_name != "instance"
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True, bias=norm_name != "instance"
        )
        self.lrelu = get_act_layer(name=act_name)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)

        self.norm_name = norm_name
        if norm_name == "instance":
            self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
            self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride, dropout=dropout, conv_only=True, bias=norm_name != "instance"
            )
            if norm_name == "instance":
                self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)


    def load_state_dict(self, state_dict, strict=True):
        if self.norm_name != "instance":
            for name, module in self.named_modules():
                if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d):
                    norm_name = name.replace("conv", "norm").split('.')[0]

                    conv_weight = state_dict[name + ".weight"]
                    conv_bias = state_dict.get(name + ".bias", torch.zeros(conv_weight.shape[0]))
                    norm_weight = state_dict.pop(norm_name + ".weight", None)
                    norm_bias = state_dict.pop(norm_name + ".bias", None)
                    norm_mean = state_dict.pop(norm_name + ".running_mean", None)
                    norm_var = state_dict.pop(norm_name + ".running_var", None)
                    eps = 1e-5
                    
                    scale = norm_weight / torch.sqrt(norm_var + eps)
                    conv_weight = conv_weight * scale.view(-1, 1, 1, 1, 1)
                    conv_bias = (conv_bias - norm_mean) * scale + norm_bias
                    
                    # Replace in state_dict
                    state_dict[name + ".weight"] = conv_weight
                    state_dict[name + ".bias"] = conv_bias
                    
                    state_dict.pop(norm_name + ".num_batches_tracked", None)
        
        # Call the original load_state_dict
        super().load_state_dict(state_dict, strict)


    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        if self.norm_name == "instance":
            out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        if self.norm_name == "instance":
            out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:

    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


if __name__ == '__main__':
    # Initialize the original and fused blocks
    spatial_dims = 3
    in_channels = 3
    out_channels = 32
    kernel_size = 3
    stride = 1
    input_shape = (1, in_channels, 32, 32, 32)  # Example input shape for 3D data

    original = UnetResBlock(spatial_dims, in_channels, out_channels, kernel_size, stride, 'batch')
    fused = UnetResBlockFused(spatial_dims, in_channels, out_channels, kernel_size, stride, 'batch')

    # Set both models to evaluation mode
    original.eval()
    fused.eval()

    # # Generate random weights for the original block
    for name, param in original.named_parameters():
        param.data = torch.randn_like(param)

    # # Properly initialize BatchNorm running statistics
    for name, buffer in original.named_buffers():
        if "running_mean" in name:
            buffer.data = torch.randn_like(buffer) * 0.1  # Small random values for running_mean
        elif "running_var" in name:
            buffer.data = torch.abs(torch.randn_like(buffer)) + 1e-5  # Positive values for running_var

    # Generate random input
    random_input = torch.rand(input_shape)

    # Fuse the weights from the original block into the fused block
    state_dict = original.state_dict()
    print(state_dict.keys())
    breakpoint()
    fused.load_state_dict(state_dict)

    # Pass the random input through both blocks
    output_original = original(random_input)
    output_fused = fused(random_input)
    # Compute the difference between the outputs
    abs_diff = torch.abs(output_original - output_fused)
    print(abs_diff.max(), abs_diff.mean())
    print(torch.allclose(output_original, output_fused, atol=1e-5, rtol=1e-3))
