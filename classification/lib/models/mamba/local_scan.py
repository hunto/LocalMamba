
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def triton_local_scan(
    x, # x point (B, C, H, W) or (B, C, L)
    y, # y point (B, C, H, W) or (B, C, L)
    K: tl.constexpr,  # window size
    flip: tl.constexpr, # whether to flip the tokens
    BC: tl.constexpr,  # number of channels in each program
    BH: tl.constexpr,  # number of heights in each program
    BW: tl.constexpr,  # number of width in each program
    DC: tl.constexpr,  # original channels
    DH: tl.constexpr,  # original height
    DW: tl.constexpr,  # original width
    NH: tl.constexpr,  # number of programs on height
    NW: tl.constexpr,  # number of programs on width
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)  # program id of hw axis, c axis, batch axis
    i_h, i_w = (i_hw // NW), (i_hw % NW)  # program idx of h and w
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]  # [BH, BW]
    _for_C = min(DC - i_c * BC, BC)  # valid number of c in the program

    _tmp0 = i_c * BC * DH * DW  # start offset of this program
    _tmp1 = DC * DH * DW  # n_elements in one batch
    _tmp2 = _tmp0 + i_h * BH * DW  + tl.arange(0, BH)[:, None] * DW + i_w * BW + tl.arange(0, BW)[None, :]  # offsets of elements in this program
    
    p_x = x + i_b * _tmp1 + _tmp2

    _i = (tl.arange(0, BH) + BH * i_h)[:, None]
    _j = (tl.arange(0, BW) + BW * i_w)[None, :]
    _c_offset = ((DW // K) * (_i // K) + (_j // K)) * K * K + (_i % K) * K + _j % K
    if flip:
        _c_offset = DH * DW - _c_offset - 1

    p_y = y + i_b * _tmp1 + _tmp0 + _c_offset
    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        tl.store(p_y + _idx, _x, mask=_mask_hw)
    tl.debug_barrier()


@triton.jit
def triton_local_reverse(
    x, # x point (B, C, H, W) or (B, C, L)
    y, # y point (B, C, H, W) or (B, C, L)
    K: tl.constexpr,  # window size
    flip: tl.constexpr,  # whether to flip the tokens
    BC: tl.constexpr,  # number of channels in each program
    BH: tl.constexpr,  # number of heights in each program
    BW: tl.constexpr,  # number of width in each program
    DC: tl.constexpr,  # original channels
    DH: tl.constexpr,  # original height
    DW: tl.constexpr,  # original width
    NH: tl.constexpr,  # number of programs on height
    NW: tl.constexpr,  # number of programs on width
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)  # program id of hw axis, c axis, batch axis
    i_h, i_w = (i_hw // NW), (i_hw % NW)  # program idx of h and w
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]  # [BH, BW]
    _for_C = min(DC - i_c * BC, BC)  # valid number of c in the program

    _tmp0 = i_c * BC * DH * DW  # start offset of this program
    _tmp1 = DC * DH * DW  # n_elements in one batch
    _tmp2 = _tmp0 + i_h * BH * DW  + tl.arange(0, BH)[:, None] * DW + i_w * BW + tl.arange(0, BW)[None, :]  # offsets of elements in this program
    
    p_x = x + i_b * _tmp1 + _tmp2

    _i = (tl.arange(0, BH) + BH * i_h)[:, None]
    _j = (tl.arange(0, BW) + BW * i_w)[None, :]
    _o = _i * DW + _j

    _i = _o // (K * K) // (DW // K) * K + _o % (K * K) // K
    _j = _o // (K * K) % (DW // K) * K + _o % (K * K) % K
    _c_offset = _i * DW + _j
    if flip:
        _c_offset = DH * DW - _c_offset - 1

    p_y = y + i_b * _tmp1 + _tmp0 + _c_offset
    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        tl.store(p_y + _idx, _x, mask=_mask_hw)
    tl.debug_barrier()


class LocalScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, K: int, flip: bool, H: int = None, W: int = None):
        ori_x = x
        if len(x.shape) == 4:
            B, C, H, W = x.shape
        elif len(x.shape) == 3:
            B, C, _ = x.shape
            assert H is not None and W is not None, "x must be BCHW format to infer the H W"
        else:
            raise RuntimeError(f"Unsupported shape of x: {x.shape}")
        B, C, H, W = int(B), int(C), int(H), int(W)

        ctx.ori_shape = (B, C, H, W)
        # pad tensor to make it evenly divisble by window size
        x, (H, W) = pad_tensor(x, K, H, W)
        ctx.shape = (B, C, H, W)

        BC, BH, BW = min(triton.next_power_of_2(C), 1), min(triton.next_power_of_2(H), 64), min(triton.next_power_of_2(W), 64)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        ctx.K = K
        ctx.flip = flip

        if x.stride(-1) != 1:
            x = x.contiguous()

        if len(ori_x.shape) == 4:
            y = x.new_empty((B, C, H, W))
        elif len(ori_x.shape) == 3:
            y = x.new_empty((B, C, H * W))

        triton_local_scan[(NH * NW, NC, B)](x, y, K, flip, BC, BH, BW, C, H, W, NH, NW)
        return y
    
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape

        if y.stride(-1) != 1:
            y = y.contiguous()
        if len(y.shape) == 4 or ctx.shape != ctx.ori_shape:
            x = y.new_empty((B, C, H, W))
        else:
            x = y.new_empty((B, C, H * W))

        triton_local_reverse[(NH * NW, NC, B)](y, x, ctx.K, ctx.flip, BC, BH, BW, C, H, W, NH, NW)

        if ctx.shape != ctx.ori_shape:
            _, _, ori_H, ori_W = ctx.ori_shape
            x = x[:, :, :ori_H, :ori_W]
            if len(y.shape) == 3:
                x = x.flatten(2)

        return x, None, None, None, None


class LocalReverseTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, K: int, flip: bool, H: int = None, W: int = None):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
        elif len(x.shape) == 3:
            B, C, _ = x.shape
            assert H is not None and W is not None, "x must be BCHW format to infer the H W"
        else:
            raise RuntimeError(f"Unsupported shape of x: {x.shape}")
        B, C, H, W = int(B), int(C), int(H), int(W)
        
        ctx.ori_shape = (B, C, H, W)
        # x may have been padded
        Hg, Wg = math.ceil(H / K), math.ceil(W / K)
        H, W = Hg * K, Wg * K
        ctx.shape = (B, C, H, W)

        BC, BH, BW = min(triton.next_power_of_2(C), 1), min(triton.next_power_of_2(H), 64), min(triton.next_power_of_2(W), 64)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        ctx.K = K
        ctx.flip = flip

        if x.stride(-1) != 1:
            x = x.contiguous()
        
        if len(x.shape) == 4 or ctx.ori_shape != ctx.shape:
            y = x.new_empty((B, C, H, W))
        else:
            y = x.new_empty((B, C, H * W))

        triton_local_reverse[(NH * NW, NC, B)](x, y, K, flip, BC, BH, BW, C, H, W, NH, NW)

        if ctx.ori_shape != ctx.shape:
            ori_H, ori_W = ctx.ori_shape[-2:]
            y = y[:, :, :ori_H, :ori_W]
            if len(x.shape) == 3:
                y = y.flatten(2)

        return y
    
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.ori_shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape

        _is_y_BCHW = len(y.shape) == 4

        y, (H, W) = pad_tensor(y, ctx.K, H, W)

        if y.stride(-1) != 1:
            y = y.contiguous()

        if _is_y_BCHW:
            x = y.new_empty((B, C, H, W))
        else:
            x = y.new_empty((B, C, H * W))

        triton_local_scan[(NH * NW, NC, B)](y, x, ctx.K, ctx.flip, BC, BH, BW, C, H, W, NH, NW)

        return x, None, None, None, None



def pad_tensor(x, w, H, W):
    if H % w == 0 and W % w == 0:
        return x, (H, W)
    B, C = x.shape[:2]
    if len(x.shape) == 3:
        x = x.view(B, C, H, W)
    
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    newH, newW = Hg * w, Wg * w
    x = F.pad(x, (0, newW - W, 0, newH - H))
    
    # We can skip flattening x back to BCL as the next operation
    # is triton_local_reverse / triton_local_scan, which supports
    # both BCHW and BCL inputs
    # if len(ori_x.shape) == 3:
    #     x = x.flatten(2)

    return x, (newH, newW)


"""PyTorch code for local scan and local reverse"""

def local_scan(x, w=7, H=14, W=14, h_scan=False):
    B, L, C = x.shape
    x = x.view(B, H, W, C)
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    if H % w != 0 or W % w != 0:
        newH, newW = Hg * w, Wg * w
        x = F.pad(x, (0, 0, 0, newW - W, 0, newH - H))
    if h_scan:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 3, 1, 4, 2, 5).reshape(B, -1, C)
    else:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, -1, C)
    return x

def local_reverse(x, w=7, H=14, W=14, h_scan=False):
    B, L, C = x.shape
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    if H % w != 0 or W % w != 0:
        if h_scan:
            x = x.view(B, Wg, Hg, w, w, C).permute(0, 2, 4, 1, 3, 5).reshape(B, Hg * w, Wg * w, C)
        else:
            x = x.view(B, Hg, Wg, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, Hg * w, Wg * w, C)
        x = x[:, :H, :W].reshape(B, -1, C)
    else:
        if h_scan:
            x = x.view(B, Wg, Hg, w, w, C).permute(0, 2, 4, 1, 3, 5).reshape(B, L, C)
        else:
            x = x.view(B, Hg, Wg, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, L, C)
    return x
