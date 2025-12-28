import torch
from torch import nn
import torch.nn.functional as F
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from p_network.network_architecture.layers import LayerNorm
from p_network.network_architecture.synapse.inference.inference_model.transformerblock import TransformerBlock
from p_network.network_architecture.dynunet_block import get_conv_layer, UnetResBlockFused, GathernExcite


einops, _ = optional_import("einops")

class PosEnc(nn.Module):
    def __init__(self, L_embed=10):
        super(PosEnc, self).__init__()
        self.L_embed = L_embed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = 3*(2*L_embed + 1)
        
    def embedding(self, inp):
        res = [inp]
        for l in range(self.L_embed):
            res.append(torch.sin(2**l * inp))
            res.append(torch.cos(2**l * inp))
        return torch.cat(res, dim=-1)
    
    def forward(self, pos, inp_shape):
        B = pos.shape[0]
        d, h, w = inp_shape
        
        mesh_grids = []
        
        for i in range(B):
            x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = pos[i]
            x_lb = 2*x_lb-1
            x_ub = 2*x_ub-1
            y_lb = 2*y_lb-1
            y_ub = 2*y_ub-1
            z_lb = 2*z_lb-1
            z_ub = 2*z_ub-1
            x = torch.linspace(x_lb, x_ub, d, device=self.device)
            y = torch.linspace(y_lb, y_ub, h, device=self.device)
            z = torch.linspace(z_lb, z_ub, w, device=self.device)
            
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            grid = torch.stack([X, Y, Z], dim=-1)
            
            mesh_grids.append(grid)
        
        mesh_grids = torch.stack(mesh_grids, dim=0).to(self.device, non_blocking=True)
        mesh_grids = self.embedding(mesh_grids)
        mesh_grids = mesh_grids.permute(0, 4, 1, 2, 3)
            
        return mesh_grids.contiguous()  

class SequentialWithArgs(nn.Sequential):
    def forward(self, x, posenc, global_embed, *args):
        for module in self:
            x = module(x, posenc, global_embed)  # Pass only `x` to each module
        return x

class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1, dropout=0.0, transformer_dropout_rate=0.15 ,**kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(2, 4, 4), stride=(2, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            global_embed = False
            if i >= 2:
                global_embed = True
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],  proj_size=proj_size[i], num_heads=num_heads,
                                     dropout_rate=transformer_dropout_rate, pos_embed=True, global_embed=global_embed))
            self.stages.append(SequentialWithArgs(*stage_blocks))
        self.hidden_states = []

    def forward_features(self, x, posenc):
        global_embed = False
        hidden_states = []
        posenc_prev = posenc

        x = self.downsample_layers[0](x)
        posenc_prev = F.interpolate(posenc_prev, scale_factor=(1/2,1/4,1/4))
        x = self.stages[0](x, posenc_prev, global_embed=global_embed)

        hidden_states.append(x)

        for i in range(1, 4):
            if i > 1:
                global_embed = True
            posenc_prev = F.interpolate(posenc_prev, scale_factor=(1/2,1/2,1/2))

            x = self.downsample_layers[i](x)
            x = self.stages[i](x, posenc_prev, global_embed=global_embed)
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x, posenc):
        x, hidden_states = self.forward_features(x, posenc)
        return x, hidden_states


class UnetrUpBlock(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
            global_embed: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlockFused) instead of EPA_Block (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlockFused(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
            self.gate = GathernExcite()
            self.pos_embed = nn.Conv3d(63, out_channels, 1)
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels, proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.15, pos_embed=True, global_embed=global_embed))
            self.decoder_block.append(SequentialWithArgs(*stage_blocks))

    def forward(self, inp, skip, posenc=None, global_embed=False):

        out = self.transp_conv(inp)
        out = out + skip
        if isinstance(self.decoder_block[0], SequentialWithArgs):
            out = self.decoder_block[0](out, posenc, global_embed)
        else:
            # out = out + self.pos_embed(posenc)
            out = self.decoder_block[0](out)
            out = out + self.gate(out)

        return out
