import os
import sys
from functools import partial
from typing import Callable

import torch
from torch import nn

# ===============================================
from typing import Union, Tuple, Any


def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

def selective_scan_flop_jit(inputs, outputs):
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True)
    return flops


selective_scan_flop_jit: Callable = selective_scan_flop_jit


def MambaInnerFnNoOutProj_flop_jit(inputs, outputs, layer):
    B, _, L = inputs[0].type().sizes()
    flops = 0
    # 2 MambaInnerFnNoOutProj
    # 2.1 causual conv1d
    flops += (L + layer.mixer.d_conv - 1) * layer.mixer.d_inner * layer.mixer.d_conv
    # 2.2 x_proj
    flops += L * layer.mixer.x_proj_0.in_features * layer.mixer.x_proj_0.out_features
    # 2.3 dt_proj
    flops += L * layer.mixer.dt_proj_0.in_features * layer.mixer.dt_proj_0.out_features
    D = layer.mixer.d_inner
    N = layer.mixer.d_state
    for i in range(len(layer.mixer.multi_scan.choices)):
        # A
        flops += D * L * N
        # B
        flops += D * L * N * 2
        # C
        flops += (D * N + D * N) * L
        # D
        flops += D * L
        # Z
        flops += D * L
    # merge
    attn = layer.mixer.attn
    flops += attn.global_reduce.in_features * attn.global_reduce.out_features
    # flops += attn.local_reduce.in_features * attn.local_reduce.out_features * L
    flops += attn.channel_select.in_features * attn.channel_select.out_features
    # flops += attn.spatial_select.in_features * attn.spatial_select.out_features * L
    return flops

MambaInnerFnNoOutProj_flop_jit: Callable = MambaInnerFnNoOutProj_flop_jit

supported_extra_ops={
    "aten::silu": None, # as relu is in _IGNORED_OPS
    "aten::neg": None, # as relu is in _IGNORED_OPS
    "aten::exp": None, # as relu is in _IGNORED_OPS
    "aten::flip": None, # as permute is in _IGNORED_OPS
    "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, # latter
    "prim::PythonOp.SelectiveScan": selective_scan_flop_jit, # latter
    "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit, # latter
    "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit, # latter
    "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit, # latter
    "prim::PythonOp.MambaInnerFnNoOutProj": None
}


def mmengine_flop_count(model: nn.Module = None, input_shape = (3, 224, 224), show_table=False, show_arch=False, _get_model_complexity_info=False):
    from mmengine.analysis.print_helper import is_tuple_of, FlopAnalyzer, ActivationAnalyzer, parameter_count, _format_size, complexity_stats_table, complexity_stats_str
    from mmengine.analysis.jit_analysis import _IGNORED_OPS
    from mmengine.analysis.complexity_analysis import _DEFAULT_SUPPORTED_FLOP_OPS, _DEFAULT_SUPPORTED_ACT_OPS
    from mmengine.analysis import get_model_complexity_info as mm_get_model_complexity_info
    
    # modified from mmengine.analysis
    def get_model_complexity_info(
        model: nn.Module,
        input_shape: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...],
                        None] = None,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[Any, ...],
                    None] = None,
        show_table: bool = True,
        show_arch: bool = True,
    ):
        if input_shape is None and inputs is None:
            raise ValueError('One of "input_shape" and "inputs" should be set.')
        elif input_shape is not None and inputs is not None:
            raise ValueError('"input_shape" and "inputs" cannot be both set.')

        if inputs is None:
            device = next(model.parameters()).device
            if is_tuple_of(input_shape, int):  # tuple of int, construct one tensor
                inputs = (torch.randn(1, *input_shape).to(device), )
            elif is_tuple_of(input_shape, tuple) and all([
                    is_tuple_of(one_input_shape, int)
                    for one_input_shape in input_shape  # type: ignore
            ]):  # tuple of tuple of int, construct multiple tensors
                inputs = tuple([
                    torch.randn(1, *one_input_shape).to(device)
                    for one_input_shape in input_shape  # type: ignore
                ])
            else:
                raise ValueError(
                    '"input_shape" should be either a `tuple of int` (to construct'
                    'one input tensor) or a `tuple of tuple of int` (to construct'
                    'multiple input tensors).')

        supported_extra_ops_ = supported_extra_ops.copy()
        if model.backbone.__class__.__name__ == 'MM_LocalVim':
            supported_extra_ops_["prim::PythonOp.MambaInnerFnNoOutProj"] = partial(MambaInnerFnNoOutProj_flop_jit, layer=model.backbone.layers[0])
        flop_handler = FlopAnalyzer(model, inputs).set_op_handle(**supported_extra_ops_)
        # activation_handler = ActivationAnalyzer(model, inputs)

        flops = flop_handler.total()
        # activations = activation_handler.total()
        params = parameter_count(model)['']

        flops_str = _format_size(flops)
        # activations_str = _format_size(activations)
        params_str = _format_size(params)

        if show_table:
            complexity_table = complexity_stats_table(
                flops=flop_handler,
                # activations=activation_handler,
                show_param_shapes=True,
            )
            complexity_table = '\n' + complexity_table
        else:
            complexity_table = ''

        if show_arch:
            complexity_arch = complexity_stats_str(
                flops=flop_handler,
                # activations=activation_handler,
            )
            complexity_arch = '\n' + complexity_arch
        else:
            complexity_arch = ''

        return {
            'flops': flops,
            'flops_str': flops,
            # 'activations': activations,
            # 'activations_str': activations_str,
            'params': params,
            'params_str': params,
            'out_table': complexity_table,
            'out_arch': complexity_arch
        }
    
    if _get_model_complexity_info:
        return get_model_complexity_info

    model.eval()
    analysis_results = get_model_complexity_info(
        model,
        input_shape,
        show_table=show_table,
        show_arch=show_arch,
    )
    flops = analysis_results['flops_str']
    params = analysis_results['params_str']
    # activations = analysis_results['activations_str']
    out_table = analysis_results['out_table']
    out_arch = analysis_results['out_arch']
    
    if show_arch:
        print(out_arch)
    
    if show_table:
        print(out_table)
    
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\t'
          f'Flops: {flops}\tParams: {params}\t'
        #   f'Activation: {activations}\n{split_line}'
    , flush=True)
    # print('!!!Only the backbone network is counted in FLOPs analysis.')
    # print('!!!Please be cautious if you use the results in papers. '
    #       'You may need to check if all ops are supported and verify that the '
    #       'flops computation is correct.')


def mmseg_flops(config=None, input_shape=(3, 512, 2048)):
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(config)
    cfg["work_dir"] = "/tmp"
    runner = Runner.from_cfg(cfg)
    model = runner.model.cuda()
    
    info = mmengine_flop_count(model, input_shape=input_shape)


if __name__ == "__main__":
    mmseg_flops(sys.argv[1])
