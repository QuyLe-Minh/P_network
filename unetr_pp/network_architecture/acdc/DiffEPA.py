import torch.nn as nn
import torch
import math
from unetr_pp.network_architecture.dynunet_block import UnetResBlock
from unetr_pp.network_architecture.rms_norm import RMSNorm

def lambda_init_fn(layer_idx):
    return 0.8 - 0.6*math.exp(-0.3*layer_idx)

class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            layer_idx: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPA(input_size=input_size,
                             hidden_size=hidden_size, 
                             proj_size=proj_size,
                             num_heads=num_heads, 
                             channel_attn_drop=dropout_rate,
                             spatial_attn_drop=dropout_rate, 
                             layer_idx=layer_idx)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x))

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x


class EPA(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1, layer_idx=1):
        super().__init__()
        self.num_heads = num_heads
        self.proj_size = proj_size
        self.temperature = nn.Parameter(torch.ones(num_heads*2, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads*2, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qk = nn.Linear(hidden_size, hidden_size * 2 * 2, bias=qkv_bias)
        self.vv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))        
        
        self.rms_norm = RMSNorm(hidden_size, eps=1e-5, elementwise_affine=False)
        
        self.lambda_init = lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32).normal_(mean=0,std=0.1))

    def forward(self, x):
        B, N, C = x.shape
        
        qk = self.qk(x).reshape(B, N, 2, self.num_heads*2, C // self.num_heads)
        qk = qk.permute(2, 0, 3, 1, 4)
        q_shared, k_shared = qk[0], qk[1]
        
        vv = self.vv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads)
        vv = vv.permute(2, 0, 3, 1, 4)
        v_CA, v_SA = vv[0], vv[1]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q_shared)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q_shared)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.view(B, self.num_heads, 2, C//self.num_heads, C//self.num_heads)
        attn_CA = attn_CA[:,:,0] - lambda_full * attn_CA[:,:,1]
        
        attn_CA = attn_CA.softmax(dim=-1)
        # print(attn_CA[0,0])

        x_CA = (attn_CA @ v_CA)
        x_CA = x_CA.permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2
        attn_SA = attn_SA.view(B, self.num_heads, 2, N, self.proj_size)
        attn_SA = attn_SA[:,:,0] - lambda_full * attn_SA[:,:,1]

        attn_SA = attn_SA.softmax(dim=-1)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1))
        x_SA = x_SA.permute(0, 2, 1, 3).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
