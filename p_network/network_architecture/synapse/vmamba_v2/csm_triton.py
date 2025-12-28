import torch
import warnings

WITH_TRITON = True
# WITH_TRITON = False
try:
    import triton
    import triton.language as tl
except:
    WITH_TRITON = False
    warnings.warn("Triton not installed, fall back to pytorch implements.")

# to make sure cached_property can be loaded for triton
if WITH_TRITON:
    try:
        from functools import cached_property
    except:
        warnings.warn("if you are using py37, add this line to functools.py: "
            "cached_property = lambda func: property(lru_cache()(func))")

# torch implementation ========================================
def cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, C, H, W, D = x.shape
        if scans == 0:
            y = x.new_empty((B, 12, C, H * W * D))
            y[:, 0, :, :] = x.flatten(2)
            y[:, 1, :, :] = x.permute(0, 1, 2, 4, 3).flatten(2)
            y[:, 2, :, :] = x.permute(0, 1, 3, 2, 4).flatten(2)
            y[:, 3, :, :] = x.permute(0, 1, 3, 4, 2).flatten(2)
            y[:, 4, :, :] = x.permute(0, 1, 4, 2, 3).flatten(2)
            y[:, 5, :, :] = x.permute(0, 1, 4, 3, 2).flatten(2)
            y[:, 6:, :, :] = torch.flip(y[:, 0:6, :, :], dims=[-1])
        elif scans == 1:
            y = x.view(B, 1, C, H * W * D).repeat(1, 12, 1, 1)
        elif scans == 2:
            y = x.view(B, 1, C, H * W * D).repeat(1, 6, 1, 1)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
    else:
        B, H, W, D, C = x.shape
        if scans == 0:
            y = x.new_empty((B, H * W * D, 12, C))
            y[:, :, 0, :] = x.flatten(1, 3)
            y[:, :, 1, :] = x.permute(0, 1, 3, 2, 4).flatten(1, 3)
            y[:, :, 2, :] = x.permute(0, 2, 1, 3, 4).flatten(1, 3)
            y[:, :, 3, :] = x.permute(0, 2, 3, 1, 4).flatten(1, 3)
            y[:, :, 4, :] = x.permute(0, 3, 1, 2, 4).flatten(1, 3)
            y[:, :, 5, :] = x.permute(0, 3, 2, 1, 4).flatten(1, 3)
            y[:, :, 6:, :] = torch.flip(y[:, :, :6, :], dims=[1])
        elif scans == 1:
            y = x.view(B, H * W * D, 1, C).repeat(1, 1, 12, 1)
        elif scans == 2:
            y = x.view(B, H * W * D, 1, C).repeat(1, 1, 6, 1)
            y = torch.cat([y, y.flip(dims=[1])], dim=2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y

def cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, C, H, W, D = y.shape
        y = y.view(B, K, C, -1)
        if scans == 0:
            y = y[:, :6] + y[:, 6:].flip(dims=[-1])
            y = y[:, 0] \
                + y[:, 1].view(B, C, H, D, W).permute(0, 1, 2, 4, 3).view(B, C, -1) \
                    + y[:, 2].view(B, C, W, H, D).permute(0, 1, 3, 2, 4).view(B, C, -1) \
                        + y[:, 3].view(B, C, W, D, H).permute(0, 1, 4, 2, 3).view(B, C, -1) \
                            + y[:, 4].view(B, C, D, H, W).permute(0, 1, 3, 4, 2).view(B, C, -1) \
                                + y[:, 5].view(B, C, D, W, H).permute(0, 1, 4, 3, 2).view(B, C, -1)
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y = y[:, :6] + y[:, 6:].flip(dims=[-1])
            y = y.sum(1)
    else:
        B, H, W, D, K, C = y.shape
        y = y.view(B, -1, K, C)
        if scans == 0:
            y = y[:, :, :6] + y[:, :, 6:].flip(dims=[1])
            y = y[:, :, 0] \
                + y[:, :, 1].view(B, H, D, W, C).permute(0, 1, 3, 2, 4).view(B, -1, C) \
                    + y[:, :, 2].view(B, W, H, D, C).permute(0, 2, 1, 3, 4).view(B, -1, C) \
                        + y[:, :, 3].view(B, W, D, H, C).permute(0, 3, 1, 2, 4).view(B, -1, C) \
                            + y[:, :, 4].view(B, D, H, W, C).permute(0, 2, 3, 1, 4).view(B, -1, C) \
                                + y[:, :, 5].view(B, D, W, H, C).permute(0, 3, 2, 1, 4).view(B, -1, C)
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y = y[:, :, :6] + y[:, :, 6:].flip(dims=[1])
            y = y.sum(2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()
    
    return y


def cross_scan1b1_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, _, C, H, W, D = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, 0].flatten(2),
                x[:, 1].permute(0, 1, 2, 4, 3).flatten(2),
                x[:, 2].permute(0, 1, 3, 2, 4).flatten(2),
                x[:, 3].permute(0, 1, 3, 4, 2).flatten(2),
                x[:, 4].permute(0, 1, 4, 2, 3).flatten(2),
                x[:, 5].permute(0, 1, 4, 3, 2).flatten(2),
                torch.flip(x[:, 6].flatten(2), dims=[-1]),
                torch.flip(x[:, 7].permute(0, 1, 2, 4, 3).flatten(2), dims=[-1]),
                torch.flip(x[:, 8].permute(0, 1, 3, 2, 4).flatten(2), dims=[-1]),
                torch.flip(x[:, 9].permute(0, 1, 3, 4, 2).flatten(2), dims=[-1]),
                torch.flip(x[:, 10].permute(0, 1, 4, 2, 3).flatten(2), dims=[-1]),
                torch.flip(x[:, 11].permute(0, 1, 4, 3, 2).flatten(2), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = x.flatten(3)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(2),
                x[:, 1].flatten(2),
                x[:, 2].flatten(2),
                x[:, 3].flatten(2),
                x[:, 4].flatten(2),
                x[:, 5].flatten(2),
                torch.flip(x[:, 6].flatten(2), dims=[-1]),
                torch.flip(x[:, 7].flatten(2), dims=[-1]),
                torch.flip(x[:, 8].flatten(2), dims=[-1]),
                torch.flip(x[:, 9].flatten(2), dims=[-1]),
                torch.flip(x[:, 10].flatten(2), dims=[-1]),
                torch.flip(x[:, 11].flatten(2), dims=[-1]),
            ], dim=1)
    else:
        B, H, W, D, _, C = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, :, :, :, 0].flatten(1, 3),
                x[:, :, :, :, 1].permute(0, 1, 3, 2, 4).flatten(1, 3),
                x[:, :, :, :, 2].permute(0, 2, 1, 3, 4).flatten(1, 3),
                x[:, :, :, :, 3].permute(0, 2, 3, 1, 4).flatten(1, 3),
                x[:, :, :, :, 4].permute(0, 3, 1, 2, 4).flatten(1, 3),
                x[:, :, :, :, 5].permute(0, 3, 2, 1, 4).flatten(1, 3),
                torch.flip(x[:, :, :, :, 6].flatten(1, 3), dims=[1]),
                torch.flip(x[:, :, :, :, 7].permute(0, 1, 3, 2, 4).flatten(1, 3), dims=[1]),
                torch.flip(x[:, :, :, :, 8].permute(0, 2, 1, 3, 4).flatten(1, 3), dims=[1]),
                torch.flip(x[:, :, :, :, 9].permute(0, 2, 3, 1, 4).flatten(1, 3), dims=[1]),
                torch.flip(x[:, :, :, :, 10].permute(0, 3, 1, 2, 4).flatten(1, 3), dims=[1]),
                torch.flip(x[:, :, :, :, 11].permute(0, 3, 2, 1, 4).flatten(1, 3), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = x.flatten(1, 3)
        elif scans == 2:
            y = torch.stack([
                x[:, :, :, :, 0].flatten(1, 3),
                x[:, :, :, :, 1].flatten(1, 3),
                x[:, :, :, :, 2].flatten(1, 3),
                x[:, :, :, :, 3].flatten(1, 3),
                x[:, :, :, :, 4].flatten(1, 3),
                x[:, :, :, :, 5].flatten(1, 3),
                torch.flip(x[:, :, :, :, 6].flatten(1, 3), dims=[1]),
                torch.flip(x[:, :, :, :, 7].flatten(1, 3), dims=[1]),
                torch.flip(x[:, :, :, :, 8].flatten(1, 3), dims=[1]),
                torch.flip(x[:, :, :, :, 9].flatten(1, 3), dims=[1]),
                torch.flip(x[:, :, :, :, 10].flatten(1, 3), dims=[1]),
                torch.flip(x[:, :, :, :, 11].flatten(1, 3), dims=[1]),
            ], dim=2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge1b1_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, C, H, W, D = y.shape
        y = y.view(B, K, C, -1)
        if scans == 0:
            y = torch.stack([
                y[:, 0],
                y[:, 1].view(B, -1, H, D, W).transpose(dim0=3, dim1=4).flatten(2),
                y[:, 2].view(B, -1, W, H, D).transpose(dim0=2, dim1=3).flatten(2),
                y[:, 3].view(B, -1, W, D, H).permute(0, 1, 4, 2, 3).flatten(2),
                y[:, 4].view(B, -1, D, H, W).permute(0, 1, 3, 4, 2).flatten(2),
                y[:, 5].view(B, -1, D, W, H).permute(0, 1, 4, 3, 2).flatten(2),
                torch.flip(y[:, 6], dims=[-1]),
                torch.flip(y[:, 7].view(B, -1, H, D, W).transpose(dim0=3, dim1=4).flatten(2), dims=[-1]),
                torch.flip(y[:, 8].view(B, -1, W, H, D).transpose(dim0=2, dim1=3).flatten(2), dims=[-1]),
                torch.flip(y[:, 9].view(B, -1, W, D, H).permute(0, 1, 4, 2, 3).flatten(2), dims=[-1]),
                torch.flip(y[:, 10].view(B, -1, D, H, W).permute(0, 1, 3, 4, 2).flatten(2), dims=[-1]),
                torch.flip(y[:, 11].view(B, -1, D, W, H).permute(0, 1, 4, 3, 2).flatten(2), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, 0],
                y[:, 1],
                y[:, 2],
                y[:, 3],
                y[:, 4],
                y[:, 5],
                torch.flip(y[:, 6], dims=[-1]),
                torch.flip(y[:, 7], dims=[-1]),
                torch.flip(y[:, 8], dims=[-1]),
                torch.flip(y[:, 9], dims=[-1]),
                torch.flip(y[:, 10], dims=[-1]),
                torch.flip(y[:, 11], dims=[-1]),
            ], dim=1)
    else:
        B, H, W, D, K, C = y.shape
        y = y.view(B, -1, K, C)
        if scans == 0:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1].view(B, H, D, W, C).transpose(dim0=2, dim1=3).flatten(1, 3),
                y[:, :, 2].view(B, W, H, D, C).transpose(dim0=1, dim1=2).flatten(1, 3),
                y[:, :, 3].view(B, W, D, H, C).permute(0, 3, 1, 2, 4).flatten(1, 3),
                y[:, :, 4].view(B, D, H, W, C).permute(0, 2, 3, 1, 4).flatten(1, 3),
                y[:, :, 5].view(B, D, W, H, C).permute(0, 3, 2, 1, 4).flatten(1, 3),
                torch.flip(y[:, :, 6], dims=[1]),
                torch.flip(y[:, :, 7].view(B, H, D, W, C).transpose(dim0=2, dim1=3).flatten(1, 3), dims=[1]),
                torch.flip(y[:, :, 8].view(B, W, H, D, C).transpose(dim0=1, dim1=2).flatten(1, 3), dims=[1]),
                torch.flip(y[:, :, 9].view(B, W, D, H, C).permute(0, 3, 1, 2, 4).flatten(1, 3), dims=[1]),
                torch.flip(y[:, :, 10].view(B, D, H, W, C).permute(0, 2, 3, 1, 4).flatten(1, 3), dims=[1]),
                torch.flip(y[:, :, 11].view(B, D, W, H, C).permute(0, 3, 2, 1, 4).flatten(1, 3), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1],
                y[:, :, 2],
                y[:, :, 3],
                y[:, :, 4],
                y[:, :, 5],
                torch.flip(y[:, :, 6], dims=[1]),
                torch.flip(y[:, :, 7], dims=[1]),
                torch.flip(y[:, :, 8], dims=[1]),
                torch.flip(y[:, :, 9], dims=[1]),
                torch.flip(y[:, :, 10], dims=[1]),
                torch.flip(y[:, :, 11], dims=[1])
            ], dim=2)

    if out_channel_first and (not in_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not out_channel_first) and in_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y

#TODO
class CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W, D) | (B, H, W, D, C) | (B, 12, C, H, W, D) | (B, H, W, D, 12, C)
        # y: (B, 12, C, H * W * D) | (B, H * W * D, 12, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        if one_by_one:
            B, K, C, H, W, D = x.shape
            if not in_channel_first:
                B, H, W, D, K, C = x.shape
        else:
            B, C, H, W, D = x.shape
            if not in_channel_first:
                B, H, W, D, C = x.shape
        ctx.shape = (B, C, H, W, D)

        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        y = _fn(x, in_channel_first, out_channel_first, scans)

        return y
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W, D = ctx.shape

        ys = ys.view(B, -1, C, H, W, D) if out_channel_first else ys.view(B, H, W, D, -1, C)
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)
        
        if one_by_one:
            y = y.view(B, 12, -1, H, W, D) if in_channel_first else y.view(B, H, W, D, 12, -1)
        else:
            y = y.view(B, -1, H, W, D) if in_channel_first else y.view(B, H, W, D, -1)

        return y, None, None, None, None

#TODO
class CrossMergeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W, D) | (B, H, W, D, C) | (B, 12, C, H, W, D) | (B, H, W, D, 12, C)
        # y: (B, 12, C, H * W * D) | (B, H * W * D, 12, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        B, K, C, H, W, D = ys.shape
        if not out_channel_first:
            B, H, W, D, K, C = ys.shape
        ctx.shape = (B, C, H, W, D)
        
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)

        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, c, h, w, d)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W, D = ctx.shape
    
        if not one_by_one:
            if in_channel_first:
                x = x.view(B, C, H, W, D)
            else:
                x = x.view(B, H, W, D, C)
        else:
            if in_channel_first:
                x = x.view(B, 12, C, H, W, D)
            else:
                x = x.view(B, H, W, D, 12, C)   
                     
        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        x = _fn(x, in_channel_first, out_channel_first, scans)
        x = x.view(B, 12, C, H, W, D) if out_channel_first else x.view(B, H, W, D, 12, C)
    
        return x, None, None, None, None


# triton implements ========================================
#TODO
@triton.jit
def triton_cross_scan_flex(
    x: tl.tensor, # (B, C, H, W, D) | (B, H, W, D, C) | (B, 12, C, H, W, D) | (B, H, W, D, 12, C)
    y: tl.tensor, # (B, 12, C, H, W, D) | (B, H, W, D, 12, C)
    x_layout: tl.constexpr,
    y_layout: tl.constexpr,
    operation: tl.constexpr,
    onebyone: tl.constexpr,
    scans: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    DD: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
    ND: tl.constexpr,
):
    # x_layout = 0
    # y_layout = 1 # 0 BCHWD, 1 BHWDC
    # operation = 0 # 0 scan, 1 merge
    # onebyone = 0 # 0 false, 1 true
    # scans = 0 # 0 cross scan, 1 unidirectional, 2 bidirectional

    i_hwd, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_wd = (i_hwd // (NW * ND)), (i_hwd % (NW * ND))
    i_w, i_d = (i_wd // ND), (i_wd % ND)
    
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_d = (i_d * BD + tl.arange(0, BD)) < DD
    
    # Create 3D mask for H, W, D
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _mask_hwd = _mask_hw[:, :, None] & _mask_d[None, None, :]
    
    _for_C = min(DC - i_c * BC, BC)

    # Original positions
    pos_h = (i_h * BH + tl.arange(0, BH)[:, None, None])
    pos_w = (i_w * BW + tl.arange(0, BW)[None, :, None])
    pos_d = (i_d * BD + tl.arange(0, BD)[None, None, :])
    
    # Flipped positions
    neg_h = (DH - i_h * BH - 1 - tl.arange(0, BH)[:, None, None])
    neg_w = (DW - i_w * BW - 1 - tl.arange(0, BW)[None, :, None])
    neg_d = (DD - i_d * BD - 1 - tl.arange(0, BD)[None, None, :])

    if scans == 0:
        # 12 routes = 6 permutations (HWD, HDW, WHD, WDH, DHW, DWH) + their flipped versions
        # HWD: h * W * D + w * D + d
        HWDRoute0 = pos_h * DW * DD + pos_w * DD + pos_d  # HWD
        HWDRoute1 = pos_h * DD * DW + pos_d * DW + pos_w  # HDW
        HWDRoute2 = pos_w * DH * DD + pos_h * DD + pos_d  # WHD
        HWDRoute3 = pos_w * DD * DH + pos_d * DH + pos_h  # WDH
        HWDRoute4 = pos_d * DH * DW + pos_h * DW + pos_w  # DHW
        HWDRoute5 = pos_d * DW * DH + pos_w * DH + pos_h  # DWH
        
        # Flipped versions
        HWDRoute6 = neg_h * DW * DD + neg_w * DD + neg_d  # HWD flipped
        HWDRoute7 = neg_h * DD * DW + neg_d * DW + neg_w  # HDW flipped
        HWDRoute8 = neg_w * DH * DD + neg_h * DD + neg_d  # WHD flipped
        HWDRoute9 = neg_w * DD * DH + neg_d * DH + neg_h  # WDH flipped
        HWDRoute10 = neg_d * DH * DW + neg_h * DW + neg_w  # DHW flipped
        HWDRoute11 = neg_d * DW * DH + neg_w * DH + neg_h  # DWH flipped
    elif scans == 1:
        # Use only the original HWD ordering for all routes (unidirectional)
        HWDRoute0 = pos_h * DW * DD + pos_w * DD + pos_d
        HWDRoute1 = HWDRoute0
        HWDRoute2 = HWDRoute0
        HWDRoute3 = HWDRoute0
        HWDRoute4 = HWDRoute0
        HWDRoute5 = HWDRoute0
        HWDRoute6 = HWDRoute0
        HWDRoute7 = HWDRoute0
        HWDRoute8 = HWDRoute0
        HWDRoute9 = HWDRoute0
        HWDRoute10 = HWDRoute0
        HWDRoute11 = HWDRoute0
    elif scans == 2:
        # Original and flipped but only for the main ordering (bidirectional)
        HWDRoute0 = pos_h * DW * DD + pos_w * DD + pos_d  # HWD
        HWDRoute1 = HWDRoute0
        HWDRoute2 = HWDRoute0
        HWDRoute3 = HWDRoute0
        HWDRoute4 = HWDRoute0
        HWDRoute5 = HWDRoute0
        
        # Flipped versions (only one direction flipped)
        HWDRoute6 = neg_h * DW * DD + neg_w * DD + neg_d  # HWD flipped
        HWDRoute7 = HWDRoute6
        HWDRoute8 = HWDRoute6
        HWDRoute9 = HWDRoute6
        HWDRoute10 = HWDRoute6
        HWDRoute11 = HWDRoute6

    _tmp1 = DC * DH * DW * DD

    # Pointers for y
    y_ptr_base = y + i_b * 12 * _tmp1 + (i_c * BC * DH * DW * DD if y_layout == 0 else i_c * BC)
    
    # Calculate pointers for all 12 routes
    if y_layout == 0:
        p_y1 = y_ptr_base + HWDRoute0
        p_y2 = y_ptr_base + _tmp1 + HWDRoute1
        p_y3 = y_ptr_base + 2 * _tmp1 + HWDRoute2
        p_y4 = y_ptr_base + 3 * _tmp1 + HWDRoute3
        p_y5 = y_ptr_base + 4 * _tmp1 + HWDRoute4
        p_y6 = y_ptr_base + 5 * _tmp1 + HWDRoute5
        p_y7 = y_ptr_base + 6 * _tmp1 + HWDRoute6
        p_y8 = y_ptr_base + 7 * _tmp1 + HWDRoute7
        p_y9 = y_ptr_base + 8 * _tmp1 + HWDRoute8
        p_y10 = y_ptr_base + 9 * _tmp1 + HWDRoute9
        p_y11 = y_ptr_base + 10 * _tmp1 + HWDRoute10
        p_y12 = y_ptr_base + 11 * _tmp1 + HWDRoute11
    else:
        p_y1 = y_ptr_base + HWDRoute0 * 12 * DC
        p_y2 = y_ptr_base + DC + HWDRoute1 * 12 * DC
        p_y3 = y_ptr_base + 2 * DC + HWDRoute2 * 12 * DC
        p_y4 = y_ptr_base + 3 * DC + HWDRoute3 * 12 * DC
        p_y5 = y_ptr_base + 4 * DC + HWDRoute4 * 12 * DC
        p_y6 = y_ptr_base + 5 * DC + HWDRoute5 * 12 * DC
        p_y7 = y_ptr_base + 6 * DC + HWDRoute6 * 12 * DC
        p_y8 = y_ptr_base + 7 * DC + HWDRoute7 * 12 * DC
        p_y9 = y_ptr_base + 8 * DC + HWDRoute8 * 12 * DC
        p_y10 = y_ptr_base + 9 * DC + HWDRoute9 * 12 * DC
        p_y11 = y_ptr_base + 10 * DC + HWDRoute10 * 12 * DC
        p_y12 = y_ptr_base + 11 * DC + HWDRoute11 * 12 * DC
    
    if onebyone == 0:
        # One-to-many mapping
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW * DD if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x = x_ptr_base + HWDRoute0
        else:
            p_x = x_ptr_base + HWDRoute0 * DC

        if operation == 0:
            # Scan operation: copy from x to all y routes
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW * DD if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW * DD if y_layout == 0 else idxc
                _x = tl.load(p_x + _idx_x, mask=_mask_hwd)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_hwd)
                tl.store(p_y2 + _idx_y, _x, mask=_mask_hwd)
                tl.store(p_y3 + _idx_y, _x, mask=_mask_hwd)
                tl.store(p_y4 + _idx_y, _x, mask=_mask_hwd)
                tl.store(p_y5 + _idx_y, _x, mask=_mask_hwd)
                tl.store(p_y6 + _idx_y, _x, mask=_mask_hwd)
                tl.store(p_y7 + _idx_y, _x, mask=_mask_hwd)
                tl.store(p_y8 + _idx_y, _x, mask=_mask_hwd)
                tl.store(p_y9 + _idx_y, _x, mask=_mask_hwd)
                tl.store(p_y10 + _idx_y, _x, mask=_mask_hwd)
                tl.store(p_y11 + _idx_y, _x, mask=_mask_hwd)
                tl.store(p_y12 + _idx_y, _x, mask=_mask_hwd)
        elif operation == 1:
            # Merge operation: sum all y routes and store in x
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW * DD if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW * DD if y_layout == 0 else idxc
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hwd)
                _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hwd)
                _y3 = tl.load(p_y3 + _idx_y, mask=_mask_hwd)
                _y4 = tl.load(p_y4 + _idx_y, mask=_mask_hwd)
                _y5 = tl.load(p_y5 + _idx_y, mask=_mask_hwd)
                _y6 = tl.load(p_y6 + _idx_y, mask=_mask_hwd)
                _y7 = tl.load(p_y7 + _idx_y, mask=_mask_hwd)
                _y8 = tl.load(p_y8 + _idx_y, mask=_mask_hwd)
                _y9 = tl.load(p_y9 + _idx_y, mask=_mask_hwd)
                _y10 = tl.load(p_y10 + _idx_y, mask=_mask_hwd)
                _y11 = tl.load(p_y11 + _idx_y, mask=_mask_hwd)
                _y12 = tl.load(p_y12 + _idx_y, mask=_mask_hwd)
                tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4 + _y5 + _y6 + _y7 + _y8 + _y9 + _y10 + _y11 + _y12, mask=_mask_hwd)
    else:
        # One-to-one mapping
        x_ptr_base = x + i_b * 12 * _tmp1 + (i_c * BC * DH * DW * DD if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x1 = x_ptr_base + HWDRoute0
            p_x2 = p_x1 + _tmp1
            p_x3 = p_x2 + _tmp1
            p_x4 = p_x3 + _tmp1
            p_x5 = p_x4 + _tmp1
            p_x6 = p_x5 + _tmp1
            p_x7 = p_x6 + _tmp1
            p_x8 = p_x7 + _tmp1
            p_x9 = p_x8 + _tmp1
            p_x10 = p_x9 + _tmp1
            p_x11 = p_x10 + _tmp1
            p_x12 = p_x11 + _tmp1  
        else:
            p_x1 = x_ptr_base + HWDRoute0 * 12 * DC
            p_x2 = p_x1 + DC
            p_x3 = p_x2 + DC
            p_x4 = p_x3 + DC
            p_x5 = p_x4 + DC
            p_x6 = p_x5 + DC
            p_x7 = p_x6 + DC
            p_x8 = p_x7 + DC
            p_x9 = p_x8 + DC
            p_x10 = p_x9 + DC
            p_x11 = p_x10 + DC
            p_x12 = p_x11 + DC
    
        if operation == 0:
            # Copy from each x route to corresponding y route
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW * DD if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW * DD if y_layout == 0 else idxc
                tl.store(p_y1 + _idx_y, tl.load(p_x1 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_y2 + _idx_y, tl.load(p_x2 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_y3 + _idx_y, tl.load(p_x3 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_y4 + _idx_y, tl.load(p_x4 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_y5 + _idx_y, tl.load(p_x5 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_y6 + _idx_y, tl.load(p_x6 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_y7 + _idx_y, tl.load(p_x7 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_y8 + _idx_y, tl.load(p_x8 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_y9 + _idx_y, tl.load(p_x9 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_y10 + _idx_y, tl.load(p_x10 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_y11 + _idx_y, tl.load(p_x11 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_y12 + _idx_y, tl.load(p_x12 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
        else:
            # Copy from each y route to corresponding x route
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW * DD if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW * DD if y_layout == 0 else idxc
                tl.store(p_x1 + _idx_x, tl.load(p_y1 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_x2 + _idx_x, tl.load(p_y2 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_x3 + _idx_x, tl.load(p_y3 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_x4 + _idx_x, tl.load(p_y4 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_x5 + _idx_x, tl.load(p_y5 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_x6 + _idx_x, tl.load(p_y6 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_x7 + _idx_x, tl.load(p_y7 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_x8 + _idx_x, tl.load(p_y8 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_x9 + _idx_x, tl.load(p_y9 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_x10 + _idx_x, tl.load(p_y10 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_x11 + _idx_x, tl.load(p_y11 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
                tl.store(p_x12 + _idx_x, tl.load(p_y12 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
# @triton.jit
# def triton_cross_scan_flex(
#     x: tl.tensor, # (B, C, H, W, D) | (B, H, W, D, C) | (B, 12, C, H, W, D) | (B, H, W, D, 12, C)
#     y: tl.tensor, # (B, 12, C, H, W, D) | (B, H, W, D, 12, C)
#     x_layout: tl.constexpr,
#     y_layout: tl.constexpr,
#     operation: tl.constexpr,
#     onebyone: tl.constexpr,
#     scans: tl.constexpr,
#     BC: tl.constexpr,   #Block channels
#     BH: tl.constexpr,   # Block height
#     BW: tl.constexpr,
#     BD: tl.constexpr,
#     DC: tl.constexpr,   # Data channels
#     DH: tl.constexpr,
#     DW: tl.constexpr,
#     DD: tl.constexpr,
#     NH: tl.constexpr,   #Num blocks along height
#     NW: tl.constexpr,
#     ND: tl.constexpr,
# ):
#     # x_layout = 0
#     # y_layout = 1 # 0 BCHW, 1 BHWC
#     # operation = 0 # 0 scan, 1 merge
#     # onebyone = 0 # 0 false, 1 true
#     # scans = 0 # 0 cross scan, 1 unidirectional, 2 bidirectional

#     i_hwd, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
#     i_h, i_w, i_d = ((i_hwd // NW) % NH), (i_hwd % NW), (i_hwd // (NW * NH))
    
#     _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
#     _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
#     _mask_d = (i_d * BD + tl.arange(0, BD)) < DD
#     _mask_hwd = _mask_h[:, None, None] & _mask_w[None, :, None] & _mask_d[None, None, :]
#     _for_C = min(DC - i_c * BC, BC)

#     pos_h = (i_h * BH + tl.arange(0, BH)[:, None, None])    #index of h from bottom to top in the whole, shape: (BH, 1, 1)
#     pos_w = (i_w * BW + tl.arange(0, BW)[None, :, None])
#     pos_d = (i_d * BD + tl.arange(0, BD)[None, None, :])
#     neg_h = (DH - i_h * BH - 1 - tl.arange(0, BH)[:, None, None])   #index of h from top to bottom in the whole
#     neg_w = (DW - i_w * BW - 1 - tl.arange(0, BW)[None, :, None])
#     neg_d = (DD - i_d * BD - 1 - tl.arange(0, BD)[None, None, :])
    
#     #TODO
#     if scans == 0:
#         HWDRoute0 = pos_h * DW * DD + pos_w * DD + pos_d    #HWD in row major order
#         HWDRoute1 = pos_h * DW * DD + pos_d * DW + pos_w    #HDW in row major order
#         HWDRoute2 = pos_w * DH * DD + pos_h * DD + pos_d
#         HWDRoute3 = pos_w * DH * DD + pos_d * DH + pos_h
#         HWDRoute4 = pos_d * DW * DH + pos_h * DW + pos_w
#         HWDRoute5 = pos_d * DW * DH + pos_w * DH + pos_h
#         HWDRoute6 = neg_h * DW * DD + neg_w * DD + neg_d
#         HWDRoute7 = neg_h * DW * DD + neg_d * DW + neg_w
#         HWDRoute8 = neg_w * DH * DD + neg_h * DD + neg_d
#         HWDRoute9 = neg_w * DH * DD + neg_d * DH + neg_h
#         HWDRoute10 = neg_d * DW * DH + neg_h * DW + neg_w
#         HWDRoute11 = neg_d * DW * DH + neg_w * DH + neg_h
        
#     elif scans == 1:
#         # none; none; none; none;
#         HWDRoute0 = pos_h * DW * DD + pos_w * DD + pos_d 
#         HWDRoute1 = HWDRoute0
#         HWDRoute2 = HWDRoute0
#         HWDRoute3 = HWDRoute0
#         HWDRoute4 = HWDRoute0
#         HWDRoute5 = HWDRoute0
#         HWDRoute6 = HWDRoute0
#         HWDRoute7 = HWDRoute0
#         HWDRoute8 = HWDRoute0
#         HWDRoute9 = HWDRoute0
#         HWDRoute10 = HWDRoute0
#         HWDRoute11 = HWDRoute0
        
#     elif scans == 2:
#         # none; none; flip; flip;
#         HWDRoute0 = pos_h * DW * DD + pos_w * DD + pos_d 
#         HWDRoute1 = HWDRoute0
#         HWDRoute2 = HWDRoute0
#         HWDRoute3 = HWDRoute0
#         HWDRoute4 = HWDRoute0
#         HWDRoute5 = HWDRoute0
#         HWDRoute6 = neg_h * DW * DD + neg_w * DD + neg_d
#         HWDRoute7 = HWDRoute6
#         HWDRoute8 = HWDRoute6
#         HWDRoute9 = HWDRoute6
#         HWDRoute10 = HWDRoute6
#         HWDRoute11 = HWDRoute6    

#     _tmp1 = DC * DH * DW * DD

#     y_ptr_base = y + i_b * 12 * _tmp1 + (i_c * BC * DH * DW * DD if y_layout == 0 else i_c * BC)
#     if y_layout == 0:
#         p_y1 = y_ptr_base + HWDRoute0
#         p_y2 = y_ptr_base + _tmp1 + HWDRoute1
#         p_y3 = y_ptr_base + 2 * _tmp1 + HWDRoute2
#         p_y4 = y_ptr_base + 3 * _tmp1 + HWDRoute3
#         p_y5 = y_ptr_base + 4 * _tmp1 + HWDRoute4
#         p_y6 = y_ptr_base + 5 * _tmp1 + HWDRoute5
#         p_y7 = y_ptr_base + 6 * _tmp1 + HWDRoute6
#         p_y8 = y_ptr_base + 7 * _tmp1 + HWDRoute7
#         p_y9 = y_ptr_base + 8 * _tmp1 + HWDRoute8
#         p_y10 = y_ptr_base + 9 * _tmp1 + HWDRoute9
#         p_y11 = y_ptr_base + 10 * _tmp1 + HWDRoute10
#         p_y12 = y_ptr_base + 11 * _tmp1 + HWDRoute11
#     else:
#         p_y1 = y_ptr_base + HWDRoute0 * 12 * DC
#         p_y2 = y_ptr_base + DC + HWDRoute1 * 12 * DC
#         p_y3 = y_ptr_base + 2 * DC + HWDRoute2 * 12 * DC
#         p_y4 = y_ptr_base + 3 * DC + HWDRoute3 * 12 * DC
#         p_y5 = y_ptr_base + 4 * DC + HWDRoute4 * 12 * DC
#         p_y6 = y_ptr_base + 5 * DC + HWDRoute5 * 12 * DC
#         p_y7 = y_ptr_base + 6 * DC + HWDRoute6 * 12 * DC
#         p_y8 = y_ptr_base + 7 * DC + HWDRoute7 * 12 * DC
#         p_y9 = y_ptr_base + 8 * DC + HWDRoute8 * 12 * DC
#         p_y10 = y_ptr_base + 9 * DC + HWDRoute9 * 12 * DC
#         p_y11 = y_ptr_base + 10 * DC + HWDRoute10 * 12 * DC
#         p_y12 = y_ptr_base + 11 * DC + HWDRoute11 * 12 * DC     
    
#     if onebyone == 0:
#         x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW * DD if x_layout == 0 else i_c * BC)
#         if x_layout == 0:
#             p_x = x_ptr_base + HWDRoute0
#         else:
#             p_x = x_ptr_base + HWDRoute0 * DC

#         #TODO: Check if need idx_z
#         if operation == 0:
#             for idxc in range(_for_C):
#                 _idx_x = idxc * DH * DW * DD if x_layout == 0 else idxc
#                 _idx_y = idxc * DH * DW * DD if y_layout == 0 else idxc
#                 _x = tl.load(p_x + _idx_x, mask=_mask_hwd)
#                 tl.store(p_y1 + _idx_y, _x, mask=_mask_hwd)
#                 tl.store(p_y2 + _idx_y, _x, mask=_mask_hwd)
#                 tl.store(p_y3 + _idx_y, _x, mask=_mask_hwd)
#                 tl.store(p_y4 + _idx_y, _x, mask=_mask_hwd)
#                 tl.store(p_y5 + _idx_y, _x, mask=_mask_hwd)
#                 tl.store(p_y6 + _idx_y, _x, mask=_mask_hwd)
#                 tl.store(p_y7 + _idx_y, _x, mask=_mask_hwd)
#                 tl.store(p_y8 + _idx_y, _x, mask=_mask_hwd)
#                 tl.store(p_y9 + _idx_y, _x, mask=_mask_hwd)
#                 tl.store(p_y10 + _idx_y, _x, mask=_mask_hwd)
#                 tl.store(p_y11 + _idx_y, _x, mask=_mask_hwd)
#                 tl.store(p_y12 + _idx_y, _x, mask=_mask_hwd)
#         elif operation == 1:
#             for idxc in range(_for_C):
#                 _idx_x = idxc * DH * DW * DD if x_layout == 0 else idxc
#                 _idx_y = idxc * DH * DW * DD if y_layout == 0 else idxc
#                 _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hwd)
#                 _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hwd)
#                 _y3 = tl.load(p_y3 + _idx_y, mask=_mask_hwd)
#                 _y4 = tl.load(p_y4 + _idx_y, mask=_mask_hwd)
#                 _y5 = tl.load(p_y5 + _idx_y, mask=_mask_hwd)
#                 _y6 = tl.load(p_y6 + _idx_y, mask=_mask_hwd)
#                 _y7 = tl.load(p_y7 + _idx_y, mask=_mask_hwd)
#                 _y8 = tl.load(p_y8 + _idx_y, mask=_mask_hwd)
#                 _y9 = tl.load(p_y9 + _idx_y, mask=_mask_hwd)
#                 _y10 = tl.load(p_y10 + _idx_y, mask=_mask_hwd)
#                 _y11 = tl.load(p_y11 + _idx_y, mask=_mask_hwd)
#                 _y12 = tl.load(p_y12 + _idx_y, mask=_mask_hwd)
#                 tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4 + _y5 + _y6 + _y7 + _y8 + _y9 + _y10 + _y11 + _y12, mask=_mask_hwd)

#     else:
#         x_ptr_base = x + i_b * 12 * _tmp1 + (i_c * BC * DH * DW * DD if x_layout == 0 else i_c * BC)
#         if x_layout == 0:
#             p_x1 = x_ptr_base + HWDRoute0
#             p_x2 = p_x1 + _tmp1
#             p_x3 = p_x2 + _tmp1
#             p_x4 = p_x3 + _tmp1
#             p_x5 = p_x4 + _tmp1
#             p_x6 = p_x5 + _tmp1
#             p_x7 = p_x6 + _tmp1
#             p_x8 = p_x7 + _tmp1
#             p_x9 = p_x8 + _tmp1
#             p_x10 = p_x9 + _tmp1
#             p_x11 = p_x10 + _tmp1
#             p_x12 = p_x11 + _tmp1
            
#         else:
#             p_x1 = x_ptr_base + HWDRoute0 * 12 * DC
#             p_x2 = p_x1 + DC
#             p_x3 = p_x2 + DC
#             p_x4 = p_x3 + DC
#             p_x5 = p_x4 + DC
#             p_x6 = p_x5 + DC
#             p_x7 = p_x6 + DC
#             p_x8 = p_x7 + DC
#             p_x9 = p_x8 + DC
#             p_x10 = p_x9 + DC
#             p_x11 = p_x10 + DC
#             p_x12 = p_x11 + DC
              
    
#         if operation == 0:
#             for idxc in range(_for_C):
#                 _idx_x = idxc * DH * DW * DD if x_layout == 0 else idxc
#                 _idx_y = idxc * DH * DW * DD if y_layout == 0 else idxc
#                 tl.store(p_y1 + _idx_y, tl.load(p_x1 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_y2 + _idx_y, tl.load(p_x2 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_y3 + _idx_y, tl.load(p_x3 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_y4 + _idx_y, tl.load(p_x4 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_y5 + _idx_y, tl.load(p_x5 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_y6 + _idx_y, tl.load(p_x6 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_y7 + _idx_y, tl.load(p_x7 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_y8 + _idx_y, tl.load(p_x8 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_y9 + _idx_y, tl.load(p_x9 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_y10 + _idx_y, tl.load(p_x10 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_y11 + _idx_y, tl.load(p_x11 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_y12 + _idx_y, tl.load(p_x12 + _idx_x, mask=_mask_hwd), mask=_mask_hwd)
                
#         else:
#             for idxc in range(_for_C):
#                 _idx_x = idxc * DH * DW * DD if x_layout == 0 else idxc
#                 _idx_y = idxc * DH * DW * DD if y_layout == 0 else idxc
#                 # Corrected: Load from y and store into x for operation 1 (merge)
#                 tl.store(p_x1 + _idx_x, tl.load(p_y1 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_x2 + _idx_x, tl.load(p_y2 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_x3 + _idx_x, tl.load(p_y3 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_x4 + _idx_x, tl.load(p_y4 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_x5 + _idx_x, tl.load(p_y5 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_x6 + _idx_x, tl.load(p_y6 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_x7 + _idx_x, tl.load(p_y7 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_x8 + _idx_x, tl.load(p_y8 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_x9 + _idx_x, tl.load(p_y9 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_x10 + _idx_x, tl.load(p_y10 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_x11 + _idx_x, tl.load(p_y11 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)
#                 tl.store(p_x12 + _idx_x, tl.load(p_y12 + _idx_y, mask=_mask_hwd), mask=_mask_hwd)

#         print("DONEEEE")

#TODO
class CrossScanTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if one_by_one:
            if in_channel_first:
                B, _, C, H, W, D = x.shape
            else:
                B, H, W, D, _, C = x.shape
        else:
            if in_channel_first:
                B, C, H, W, D = x.shape
            else:
                B, H, W, D, C = x.shape
        B, C, H, W, D = int(B), int(C), int(H), int(W), int(D)
        BC, BH, BW, BD = 1, 32, 32, 32
        NH, NW, ND, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(D, BD), triton.cdiv(C, BC)
        
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W, D)
        ctx.triton_shape = (BC, BH, BW, BD, NC, NH, NW, ND)

        y = x.new_empty((B, 12, C, H * W * D)) if out_channel_first else x.new_empty((B, H * W * D, 12, C))
        print(NH * NW * ND, NC, B)
        triton_cross_scan_flex[(NH * NW * ND, NC, B)](
            x.contiguous(), y, 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans, 
            BC, BH, BW, BD, C, H, W, D, NH, NW, ND
        )
        return y
        
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W, D = ctx.shape
        BC, BH, BW, BD, NC, NH, NW, ND = ctx.triton_shape
        if one_by_one:
            x = y.new_empty((B, 12, C, H, W, D)) if in_channel_first else y.new_empty((B, H, W, D, 12, C))
        else:
            x = y.new_empty((B, C, H, W, D)) if in_channel_first else y.new_empty((B, H, W, D, C))
        
        triton_cross_scan_flex[(NH * NW * ND, NC, B)](
            x, y.contiguous(), 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, BD, C, H, W, D, NH, NW, ND
        )
        return x, None, None, None, None

#TODO
class CrossMergeTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if out_channel_first:
            B, _, C, H, W, D = y.shape
        else:
            B, H, W, D, _, C = y.shape
        B, C, H, W, D = int(B), int(C), int(H), int(W), int(D)
        BC, BH, BW, BD = 1, 32, 32, 32
        NH, NW, ND, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(D, BD), triton.cdiv(C, BC)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W, D)
        ctx.triton_shape = (BC, BH, BW, BD, NC, NH, NW, ND)
        if one_by_one:
            x = y.new_empty((B, 12, C, H * W * D)) if in_channel_first else y.new_empty((B, H * W * D, 12, C))
        else:
            x = y.new_empty((B, C, H * W * D)) if in_channel_first else y.new_empty((B, H * W * D, C))
        triton_cross_scan_flex[(NH * NW * ND, NC, B)](
            x, y.contiguous(), 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, BD, C, H, W, D, NH, NW, ND
        )
        return x
        
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W, D = ctx.shape
        BC, BH, BW, BD, NC, NH, NW, ND = ctx.triton_shape
        y = x.new_empty((B, 12, C, H, W, D)) if out_channel_first else x.new_empty((B, H, W, D, 12, C))
        triton_cross_scan_flex[(NH * NW * ND, NC, B)](
            x.contiguous(), y, 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans,
            BC, BH, BW, BD, C, H, W, D, NH, NW, ND
        )
        return y, None, None, None, None, None

#TODO
# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_scan_fn(x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    # x: (B, C, H, W, D) | (B, H, W, D, C) | (B, 12, C, H, W, D) | (B, H, W, D, 12, C)
    # y: (B, 12, C, L) | (B, L, 12, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    CSF = CrossScanTritonF if WITH_TRITON and x.is_cuda and (not force_torch) else CrossScanF
    with torch.cuda.device(x.device):
        return CSF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)

#TODO
# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_merge_fn(y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    # y: (B, 12, C, L) | (B, L, 12, C)
    # x: (B, C, H * W * D) | (B, H * W * D, C) | (B, 12, C, H * W * D) | (B, H * W * D, 12, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    CMF = CrossMergeTritonF if WITH_TRITON and y.is_cuda and (not force_torch) else CrossMergeF
    with torch.cuda.device(y.device):
        return CMF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)


# checks =================================================================
#TODO
class CHECK:
    def check_csm_triton():
        B, C, H, W, D = 2, 32, 32, 32, 32
        dtype=torch.float16
        dtype=torch.float32
        x = torch.randn((B, C, H, W, D), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
        y = torch.randn((B, 12, C, H, W, D), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
        x1 = x.clone().detach().requires_grad_(True)
        y1 = y.clone().detach().requires_grad_(True)

        def cross_scan(x: torch.Tensor):
            B, C, H, W, D = x.shape
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
            return xs
        
        def cross_merge(out_y: torch.Tensor):
            B, K, C, H, W, D = out_y.shape
            L = H * W * D
            out_y = out_y.view(B, K, C, L)
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
            return out_y[:, 0] + inv_y[:, 0] + hdw_y + whd_y + wdh_y + dhw_y + dwh_y + inv_hdw_y + inv_whd_y + inv_wdh_y + inv_dhw_y + inv_dwh_y

        def cross_scan_1b1(x: torch.Tensor):
            B, K, C, H, W, D = x.shape
            L = H * W * D
            xs = torch.stack([
                x[:, 0].view(B, C, L),
                x[:, 1].permute(0, 1, 2, 4, 3).contiguous().view(B, C, L),
                x[:, 2].permute(0, 1, 3, 2, 4).contiguous().view(B, C, L),
                x[:, 3].permute(0, 1, 3, 4, 2).contiguous().view(B, C, L),
                x[:, 4].permute(0, 1, 4, 2, 3).contiguous().view(B, C, L),
                x[:, 5].permute(0, 1, 4, 3, 2).contiguous().view(B, C, L),
                torch.flip(x[:, 0].view(B, C, L), dims=[-1]),
                torch.flip(x[:, 1].permute(0, 1, 2, 4, 3).contiguous().view(B, C, L), dims=[-1]),
                torch.flip(x[:, 2].permute(0, 1, 3, 2, 4).contiguous().view(B, C, L), dims=[-1]),
                torch.flip(x[:, 3].permute(0, 1, 3, 4, 2).contiguous().view(B, C, L), dims=[-1]),
                torch.flip(x[:, 4].permute(0, 1, 4, 2, 3).contiguous().view(B, C, L), dims=[-1]),
                torch.flip(x[:, 5].permute(0, 1, 4, 3, 2).contiguous().view(B, C, L), dims=[-1])
            ], dim=1).view(B, 12, C, L)
            return xs
        
        def unidi_scan(x):
            B, C, H, W, D = x.shape
            x = x.view(B, 1, C, H * W * D).repeat(1, 12, 1, 1)
            return x
        
        def unidi_merge(ys):
            B, K, C, H, W, D = ys.shape
            return ys.view(B, 12, -1, H * W  * D).sum(1)

        def bidi_scan(x):
            B, C, H, W, D = x.shape
            x = x.view(B, 1, C, H * W * D).repeat(1, 6, 1, 1)
            x = torch.cat([x, x.flip(dims=[-1])], dim=1)
            return x
        
        def bidi_merge(ys):
            B, K, C, H, W, D = ys.shape
            ys = ys.view(B, K, C, -1)
            ys = ys[:, :6] + ys[:, 6:].flip(dims=[-1])
            return ys.contiguous().sum(1)

        if True:
            res0 = triton.testing.do_bench(lambda :cross_scan(x))
            res1 = triton.testing.do_bench(lambda :cross_scan_fn(x, True, True, False))
            # res2 = triton.testing.do_bench(lambda :CrossScanTriton.apply(x))
            res3 = triton.testing.do_bench(lambda :cross_merge(y))
            res4 = triton.testing.do_bench(lambda :cross_merge_fn(y, True, True, False))
            # res5 = triton.testing.do_bench(lambda :CrossMergeTriton.apply(y))
            # print(res0, res1, res2, res3, res4, res5)
            print(res0, res1, res3, res4)
            res0 = triton.testing.do_bench(lambda :cross_scan(x).sum().backward())
            res1 = triton.testing.do_bench(lambda :cross_scan_fn(x, True, True, False).sum().backward())
            # res2 = triton.testing.do_bench(lambda :CrossScanTriton.apply(x).sum().backward())
            res3 = triton.testing.do_bench(lambda :cross_merge(y).sum().backward())
            res4 = triton.testing.do_bench(lambda :cross_merge_fn(y, True, True, False).sum().backward())
            # res5 = triton.testing.do_bench(lambda :CrossMergeTriton.apply(y).sum().backward())
            # print(res0, res1, res2, res3, res4, res5)
            print(res0, res1, res3, res4)

        print("test cross scan")
        for (cs0, cm0, cs1, cm1) in [
            # channel_first -> channel_first
            (cross_scan, cross_merge, cross_scan_fn, cross_merge_fn),
            (unidi_scan, unidi_merge, lambda x: cross_scan_fn(x, scans=1), lambda x: cross_merge_fn(x, scans=1)),
            (bidi_scan, bidi_merge, lambda x: cross_scan_fn(x, scans=2), lambda x: cross_merge_fn(x, scans=2)),
            
            # flex: BLC->BCL; BCL->BLC; BLC->BLC;
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x.permute(0, 2, 3, 4, 1), in_channel_first=False), lambda x: cross_merge_fn(x, in_channel_first=False).permute(0, 2, 1)),
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x, out_channel_first=False).permute(0, 2, 3, 4, 1), lambda x: cross_merge_fn(x.permute(0, 3, 4, 5, 1, 2), out_channel_first=False)),
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x.permute(0, 2, 3, 4, 1), in_channel_first=False, out_channel_first=False).permute(0, 2, 3, 4, 1), lambda x: cross_merge_fn(x.permute(0, 3, 4, 5, 1, 2), in_channel_first=False, out_channel_first=False).permute(0, 2, 1)),
            
            # previous
            # (cross_scan, cross_merge, lambda x: CrossScanTriton.apply(x), lambda x: CrossMergeTriton.apply(x)),
            # (unidi_scan, unidi_merge, lambda x: getCSM(1)[0].apply(x), lambda x: getCSM(1)[1].apply(x)),
            # (bidi_scan, bidi_merge, lambda x: getCSM(2)[0].apply(x), lambda x: getCSM(2)[1].apply(x)),
        ]:
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            o0 = cs0(x)
            o1 = cs1(x1)
            o0.backward(y.view(B, 12, C, H * W * D))
            o1.backward(y.view(B, 12, C, H * W * D))
            print((o0 - o1).abs().max())
            print((x.grad - x1.grad).abs().max())
            o0 = cm0(y)
            o1 = cm1(y1)
            o0.backward(x.view(B, C, H * W * D))
            o1.backward(x.view(B, C, H * W * D))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            print("===============", flush=True)

        print("test cross scan one by one")
        for (cs0, cs1) in [
            (cross_scan_1b1, lambda x: cross_scan_fn(x, one_by_one=True)),
            # (cross_scan_1b1, lambda x: CrossScanTriton1b1.apply(x)),
        ]:
            o0 = cs0(y)
            o1 = cs1(y1)
            o0.backward(y.view(B, 12, C, H * W * D))
            o1.backward(y.view(B, 12, C, H * W * D))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            print("===============", flush=True)

    def check_csm_scan3():
        if False:
            x = torch.arange(0, 16).view(1, 1, 4, 4).cuda()
            out1 = cross_scan_fn(x, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out2 = cross_merge_fn(out1, scans=3, force_torch=True).view(1, 1, 4, 4)
            out4 = cross_merge_fn(out1, one_by_one=True, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out3 = cross_scan_fn(out4, one_by_one=True, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out5 = cross_scan_fn(x.view(1, 4, 4, 1), in_channel_first=False, out_channel_first=False, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
            out6 = cross_merge_fn(out5, in_channel_first=False, out_channel_first=False, scans=3, force_torch=True).view(1, 4, 4, 1)
            out8 = cross_merge_fn(out5, in_channel_first=False, out_channel_first=False, one_by_one=True, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
            out7 = cross_scan_fn(out8, in_channel_first=False, out_channel_first=False, one_by_one=True, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
            print(out1.view(4, -1))
            print(out2.view(-1))
            print(out3.view(4, -1))
            print(out4.view(4, -1))
            print(out5.view(-1, 4).t())
            print(out6.view(-1))
            print(out7.view(-1, 4).t())
            print(out8.view(-1, 4).t())

        B, C, H, W, D = 2, 32, 32, 32, 32
        x = torch.randn((B, C, H, W, D)).cuda()

        for scans in [0, 1, 2, 3]:
            o1 = cross_scan_fn(x, scans=scans, force_torch=True).view(B, 12, C, H, W, D)
            print((cross_scan_fn(x, scans=scans) == cross_scan_fn(x, scans=scans, force_torch=True)).all())
            print((cross_merge_fn(o1, scans=scans) == cross_merge_fn(o1, scans=scans, force_torch=True)).all())

            kwargs = dict(in_channel_first=False, out_channel_first=False)
            o2 = o1.permute(0, 3, 4, 5, 1, 2).contiguous()
            print((cross_scan_fn(x, scans=scans, **kwargs) == cross_scan_fn(x, scans=scans, force_torch=True, **kwargs)).all())
            print((cross_merge_fn(o2, scans=scans, **kwargs) == cross_merge_fn(o2, scans=scans, force_torch=True, **kwargs)).all())            

        breakpoint()


if __name__ == "__main__":
    CHECK.check_csm_scan3()
    CHECK.check_csm_triton()



