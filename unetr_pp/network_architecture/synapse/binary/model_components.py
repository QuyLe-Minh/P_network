import tinycudann as tcnn
import json
import torch
from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from unetr_pp.network_architecture.layers import LayerNorm
from unetr_pp.network_architecture.synapse.binary.fused_former import TransformerBlock
from unetr_pp.network_architecture.dynunet_block import get_conv_layer, UnetResBlock


einops, _ = optional_import("einops")

class HashEncoding(nn.Module):
    def __init__(self, n_input_dim = 3):
        super(HashEncoding, self).__init__()
        
        with open("unetr_pp/network_architecture/synapse/bank/config_hash.json") as f:
        	config = json.load(f)
        self.encoding = tcnn.Encoding(n_input_dim, config["encoding"])

    def forward(self, inp):
        B, C, D, H, W = inp.shape
        inp = inp.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        encoded = self.encoding(inp)
        encoded = encoded.view(B, D, H, W, encoded.shape[-1]).permute(0, 4, 1, 2, 3)
        return encoded.to(torch.float32).contiguous()
            
class PosEnc(nn.Module):
    def __init__(self):
        super(PosEnc, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, pos, inp_shape):
        B = pos.shape[0]
        d, h, w = inp_shape
        
        mesh_grids = []
        
        for i in range(B):
            x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = pos[i]
            x = torch.linspace(x_lb, x_ub, d)
            y = torch.linspace(y_lb, y_ub, h)
            z = torch.linspace(z_lb, z_ub, w)
            
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            grid = torch.stack([X, Y, Z], dim=-1)
            
            mesh_grids.append(grid)
        
        mesh_grids = torch.stack(mesh_grids, dim=0).to(self.device, non_blocking=True)
        mesh_grids = mesh_grids.permute(0, 4, 1, 2, 3)
            
        return mesh_grids.contiguous()  
    
class CustomSequential(nn.Sequential):
    def forward(self, x, bank):
        for module in self:
            x = module(x, bank)
        return x

class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4],dims=[32, 64, 128, 256], proj_size =[64,64,64,32],
                 depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1, dropout=0.0, transformer_dropout_rate=0.15 ,**kwargs):
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
            stage_blocks = []
            for j in range(depths[i]):
                # stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i], num_heads=num_heads,
                #                      dropout_rate=transformer_dropout_rate, pos_embed=True))
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i], proj_size=proj_size[i], num_heads=num_heads,
                                     dropout_rate=transformer_dropout_rate, pos_embed=True, decoder=False))
            self.stages.append(CustomSequential(*stage_blocks))

        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, bank):
        hidden_states = []

        x = self.downsample_layers[0](x)

        x = self.stages[0](x, bank)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x, bank)
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x, bank):
        x, hidden_states = self.forward_features(x, bank)
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
        num_heads: int = 4,
        out_size: int = 0,
        depth: int = 3,
        conv_decoder: bool = False,
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
        self.hash = nn.Sequential(
            HashEncoding(),
            nn.Conv3d(32, out_channels, 1)
        )
        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                # stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels, num_heads=num_heads,
                #                                      dropout_rate=0.15, pos_embed=True))
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels, num_heads=num_heads,
                                                     dropout_rate=0.15, pos_embed=True, decoder=True))
            self.decoder_block.append(CustomSequential(*stage_blocks))

    def forward(self, inp, skip, posenc, n_classes, bank=None):
        out = self.transp_conv(inp)
        out = out.reshape(2, n_classes, out.shape[1], out.shape[2], out.shape[3], out.shape[4]) #B, n_classes, C, D, H, W
        skip = skip.reshape(2, 1, skip.shape[1], skip.shape[2], skip.shape[3], skip.shape[4])
        out = out + skip
        out = out.reshape(-1, out.shape[1], out.shape[2], out.shape[3], out.shape[4])
        out = out + self.hash(posenc)
        if isinstance(self.decoder_block[0], UnetResBlock):
            out = self.decoder_block[0](out)
        else:
            out = self.decoder_block[0](out, bank)
        return out
