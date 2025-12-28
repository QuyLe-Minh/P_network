import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from p_network.network_architecture.layers import LayerNorm
from p_network.network_architecture.acdc.transformerblock import TransformerBlock
from p_network.network_architecture.dynunet_block import get_conv_layer, UnetResBlock


einops, _ = optional_import("einops")

class PosEnc(nn.Module):
    def __init__(self, out_channels=1, width=256, depth=4, L_embed=10):
        super(PosEnc, self).__init__()
        self.width = width
        self.depth = depth
        self.L_embed = L_embed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.layers = nn.ModuleList()
        self.dim = 3*(2*L_embed + 1)
        self.pts_dim = self.dim
        
        self.act = nn.ReLU()
        
        for i in range(depth):
            if i==5:
                self.layers.append(self.dense(width+self.pts_dim, width, self.act))
            else:
                self.layers.append(self.dense(self.dim, width, self.act))
                self.dim = width
        
        self.layers.append(self.dense(width, out_channels))
        
    def dense(self, in_dim, out_dim, act=None):
        layers = [nn.Linear(in_dim, out_dim)]
        if act is not None:
            layers.append(act)
        return nn.Sequential(*layers)
        
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
            x = torch.linspace(x_lb, x_ub, d)
            y = torch.linspace(y_lb, y_ub, h)
            z = torch.linspace(z_lb, z_ub, w)
            
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            grid = torch.stack([X, Y, Z], dim=-1)
            
            mesh_grids.append(grid)
        
        mesh_grids = torch.stack(mesh_grids, dim=0)
        
        mesh_grids = self.embedding(mesh_grids).to(self.device, non_blocking=True)
        for i, layer in enumerate(self.layers):
            mesh_grids = layer(mesh_grids)
        
        mesh_grids = mesh_grids.permute(0, 4, 1, 2, 3)   #BDHW1->B1DHW
            
        return mesh_grids.contiguous()  

class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[16 * 40 * 40, 8 * 20 * 20, 4 * 10 * 10, 2 * 5 * 5],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1,
                 dropout=0.0, transformer_dropout_rate=0.1 ,**kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4),
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
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],
                                                     proj_size=proj_size[i], num_heads=num_heads,
                                                     dropout_rate=transformer_dropout_rate, pos_embed=True)) #, layer_idx=j
            self.stages.append(nn.Sequential(*stage_blocks))
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

    def forward_features(self, x, posenc):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        hidden_states.append(x)
        
        posenc_prev = posenc

        for i in range(1, 4):
            if posenc is not None:
                if i==1:
                    posenc_prev = F.interpolate(posenc_prev, scale_factor=(1., 1/4, 1/4))
                else:
                    posenc_prev = F.interpolate(posenc_prev, scale_factor=(1/2, 1/2, 1/2))
                x = x + posenc_prev
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
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

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block
        # (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels,
                                                     proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True)) #, layer_idx=j
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out
