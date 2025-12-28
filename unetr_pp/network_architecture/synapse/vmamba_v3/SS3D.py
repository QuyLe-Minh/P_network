from typing import Any
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from unetr_pp.network_architecture.dynunet_block import UnetResBlock

try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except:
    from csm_triton import cross_scan_fn, cross_merge_fn

try:
    from .csms6s import selective_scan_fn
except:
    from csms6s import selective_scan_fn
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear3d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear3d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class Linear3d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W, D = x.shape
        return F.conv3d(x, self.weight[:, :, None, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm3d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        # channel first
        x = x.permute(0, 2, 3, 4, 1)    # (B, C, H, W, D) -> (B, H, W, D, C)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)     # (B, H, W, D, C) -> (B, C, H, W, D)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W, D = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W, D)
        elif self.dim == 1:
            B, H, W, D, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, D, C)
        else:
            raise NotImplementedError


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs
            
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias
    
    
    @classmethod
    def init_dt_A_D_channel(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor):
        # dt proj ============================
        dt_projs = cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
        dt_projs_weight = nn.Parameter(dt_projs.weight) # (inner, rank)
        dt_projs_bias = nn.Parameter(dt_projs.bias) # (inner)
        del dt_projs
            
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner) # (D, N)
        Ds = cls.D_init(d_inner) # (D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class SS3Dv0(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_length=64,
        d_projected=64,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        dt_rank_channel="auto",
        # ======================
        dropout=0.0,
        # ======================
        seq=False,
        force_fp32=True,
        **kwargs,
    ):
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 12
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        d_inner_channel = d_projected
        self.d_projected = d_projected
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        dt_rank_channel = math.ceil(d_projected / 16) if dt_rank_channel == "auto" else dt_rank_channel

        self.forward = self.forwardv0 
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act: nn.Module = act_layer()
        self.conv3d = nn.Conv3d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_channel = nn.Linear(d_projected, (dt_rank_channel + d_state * 2), bias=False)
        self.x_proj_channel_weight = nn.Parameter(self.x_proj_channel.weight)
        del self.x_proj_channel
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj, A, D ============================
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=12,
        )
        
        self.A_logs_channel, self.Ds_channel, self.dt_projs_channel_weight, self.dt_projs_channel_bias = mamba_init.init_dt_A_D_channel(
            d_state, dt_rank_channel, d_inner_channel, dt_scale, dt_init, dt_min, dt_max, dt_init_floor
        )
        self.down_proj = nn.Linear(d_length, d_projected)
        self.up_proj = nn.Linear(d_projected, d_length)

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
    def spatial_forward(self, x: torch.Tensor, force_fp32, selective_scan):
        B, C, H, W, D = x.shape
        C, N = self.A_logs.shape
        K, C, R = self.dt_projs_weight.shape
        L = H * W * D

        xs = torch.stack([
            x.view(B, C, L),
            x.permute(0, 1, 2, 4, 3).contiguous().view(B, C, L),
            x.permute(0, 1, 3, 2, 4).contiguous().view(B, C, L),
            x.permute(0, 1, 3, 4, 2).contiguous().view(B, C, L),
            x.permute(0, 1, 4, 2, 3).contiguous().view(B, C, L),
            x.permute(0, 1, 4, 3, 2).contiguous().view(B, C, L),
            torch.flip(x.view(B, C, L), dims=[-1]),
            torch.flip(x.permute(0, 1, 2, 4, 3).contiguous().view(B, C, L), dims=[-1]),
            torch.flip(x.permute(0, 1, 3, 2, 4).contiguous().view(B, C, L), dims=[-1]),
            torch.flip(x.permute(0, 1, 3, 4, 2).contiguous().view(B, C, L), dims=[-1]),
            torch.flip(x.permute(0, 1, 4, 2, 3).contiguous().view(B, C, L), dims=[-1]),
            torch.flip(x.permute(0, 1, 4, 3, 2).contiguous().view(B, C, L), dims=[-1]),
        ], dim=1).view(B, 12, C, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.contiguous() # (b, k, d_state, l)
        Cs = Cs.contiguous() # (b, k, d_state, l)
        
        As = -self.A_logs.float().exp() # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        out_y = selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        
        inv_y = torch.flip(out_y[:, 6:], dims=[-1])
        hdw_y = out_y[:, 1].view(B, C, H, D, W).permute(0, 1, 2, 4, 3).contiguous().view(B, C, L)
        whd_y = out_y[:, 2].view(B, C, W, H, D).permute(0, 1, 3, 2, 4).contiguous().view(B, C, L)
        wdh_y = out_y[:, 3].view(B, C, W, D, H).permute(0, 1, 3, 4, 2).contiguous().view(B, C, L)
        dhw_y = out_y[:, 4].view(B, C, D, H, W).permute(0, 1, 4, 2, 3).contiguous().view(B, C, L)
        dwh_y = out_y[:, 5].view(B, C, D, W, H).permute(0, 1, 4, 3, 2).contiguous().view(B, C, L)
        inv_hdw_y = inv_y[:, 1].view(B, C, H, D, W).permute(0, 1, 2, 4, 3).contiguous().view(B, C, L)
        inv_whd_y = inv_y[:, 2].view(B, C, W, H, D).permute(0, 1, 3, 2, 4).contiguous().view(B, C, L)
        inv_wdh_y = inv_y[:, 3].view(B, C, W, D, H).permute(0, 1, 3, 4, 2).contiguous().view(B, C, L)
        inv_dhw_y = inv_y[:, 4].view(B, C, D, H, W).permute(0, 1, 4, 2, 3).contiguous().view(B, C, L)
        inv_dwh_y = inv_y[:, 5].view(B, C, D, W, H).permute(0, 1, 4, 3, 2).contiguous().view(B, C, L)
        y = out_y[:, 0] + inv_y[:, 0] + hdw_y + whd_y + wdh_y + dhw_y + dwh_y + inv_hdw_y + inv_whd_y + inv_wdh_y + inv_dhw_y + inv_dwh_y
        
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        
        return y
    
    def channel_forward(self, x: torch.Tensor, force_fp32, selective_scan):
        B, C, H, W, D = x.shape
        L, N = self.A_logs_channel.shape
        L, R = self.dt_projs_channel_weight.shape
        
        
        x = x.reshape(B, C, -1).contiguous() # (B, C, L)
        x = self.down_proj(x).transpose(-1, -2)   # (B, L_proj, C)
        
        x_dbl = torch.einsum("b l c, n l -> b n c", x, self.x_proj_channel_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=1)
        dts = torch.einsum("b r c, d r -> b d c", dts, self.dt_projs_channel_weight)
        
        dts = dts.contiguous()
        Bs = Bs.view(B, 1, N, C).repeat(1, self.d_projected, 1, 1).contiguous()
        Cs = Cs.view(B, 1, N, C).repeat(1, self.d_projected, 1, 1).contiguous()
        x = x.contiguous()
        
        As = -self.A_logs_channel.float().exp() # (l_projected, d_state)
        Ds = self.Ds_channel.float() #(L_projected)
        dt_projs_channel_bias = self.dt_projs_channel_bias.float().view(-1) #(L_projected)
        
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        
        if force_fp32:
            x, dts, Bs, Cs = to_fp32(x, dts, Bs, Cs)  
            
        y = selective_scan(
            x, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_channel_bias,
            delta_softplus=True,
            )
        
        y = self.up_proj(y.transpose(-1, -2)).transpose(-1, -2)
        return y


    def forwardv0(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        #TODO: x is in (B, H, W, D, C) format (channel last)
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d, c)
        z = self.act(z)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.conv3d(x) # (b, c, h, w, d)
        x = self.act(x)
        selective_scan = partial(selective_scan_fn, backend="mamba")
        
        B, C, H, W, D = x.shape
        y_spatial = self.spatial_forward(x, force_fp32, selective_scan)
        y_channel = self.channel_forward(x, force_fp32, selective_scan)
        
        y = y_spatial + y_channel
        y = self.out_norm(y).view(B, H, W, D, -1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out
    

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        spatial_one_dim_lenght: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v0",
        # =============================
        mlp_ratio=0.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        # =============================
        _SS3D: type = SS3Dv0,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm1 = norm_layer(hidden_dim)
            self.op = _SS3D(
                d_model=hidden_dim, 
                d_length=spatial_one_dim_lenght**3,
                d_projected=64,
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )
        
        self.gamma = nn.Parameter(1e-6 * torch.ones(1, hidden_dim), requires_grad=True)
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

        self.conv51 = UnetResBlock(3, hidden_dim, hidden_dim, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_dim, hidden_dim, 1))

    def forward(self, input: torch.Tensor):
        # input is channel first
        x1 = input.permute(0, 2, 3, 4, 1).contiguous()
        x = x1
        
        if self.ssm_branch:
            x = x + self.drop_path(self.op(self.norm1(x1)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        
        x_skip = x.permute(0, 4, 1, 2, 3).contiguous() * self.gamma # back to channel first
        x = self.conv51(x_skip)
        x = x_skip + self.conv8(x)
        return x