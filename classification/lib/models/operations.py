import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .mdconv import MDConv


OPS = OrderedDict()
OPS['id'] = lambda inp, oup, t, stride, kwargs: Identity(in_channels=inp, out_channels=oup, kernel_size=1, stride=stride, **kwargs)

'''MixConv'''
OPS['ir_mix_se'] = lambda inp, oup, t, stride, kwargs: InvertedResidualMixConv(in_channels=inp, out_channels=oup, dw_kernel_size=3,
    stride=stride, act_fn=HSwish, expand_ratio=t, se_ratio=0.25, se_gate_fn=HSigmoid, **kwargs)
OPS['ir_mix_nse'] = lambda inp, oup, t, stride, kwargs: InvertedResidualMixConv(in_channels=inp, out_channels=oup, dw_kernel_size=3, 
    stride=stride, act_fn=HSwish, expand_ratio=t, se_ratio=None, **kwargs)

'''MobileNet V2 Inverted Residual'''
OPS['ir_3x3_se'] = lambda inp, oup, t, stride, kwargs: InvertedResidual(in_channels=inp, out_channels=oup, dw_kernel_size=3, 
    stride=stride, act_fn=HSwish, expand_ratio=t, se_ratio=0.25, se_gate_fn=HSigmoid, **kwargs)
OPS['ir_5x5_se'] = lambda inp, oup, t, stride, kwargs: InvertedResidual(in_channels=inp, out_channels=oup, dw_kernel_size=5, 
    stride=stride, act_fn=HSwish, expand_ratio=t, se_ratio=0.25, se_gate_fn=HSigmoid, **kwargs)
OPS['ir_7x7_se'] = lambda inp, oup, t, stride, kwargs: InvertedResidual(in_channels=inp, out_channels=oup, dw_kernel_size=7, 
    stride=stride, act_fn=HSwish, expand_ratio=t, se_ratio=0.25, se_gate_fn=HSigmoid, **kwargs)
OPS['ir_3x3_nse'] = lambda inp, oup, t, stride, kwargs: InvertedResidual(in_channels=inp, out_channels=oup, dw_kernel_size=3, 
    stride=stride, act_fn=HSwish, expand_ratio=t, se_ratio=None, **kwargs)
OPS['ir_5x5_nse'] = lambda inp, oup, t, stride, kwargs: InvertedResidual(in_channels=inp, out_channels=oup, dw_kernel_size=5, 
    stride=stride, act_fn=HSwish, expand_ratio=t, se_ratio=None, **kwargs)
OPS['ir_7x7_nse'] = lambda inp, oup, t, stride, kwargs: InvertedResidual(in_channels=inp, out_channels=oup, dw_kernel_size=7, 
    stride=stride, act_fn=HSwish, expand_ratio=t, se_ratio=None, **kwargs)
OPS['ir_3x3'] = lambda inp, oup, t, stride, kwargs: InvertedResidual(in_channels=inp, out_channels=oup, dw_kernel_size=3, 
    stride=stride, act_fn=nn.ReLU, expand_ratio=t, se_ratio=None, **kwargs)
OPS['ir_5x5'] = lambda inp, oup, t, stride, kwargs: InvertedResidual(in_channels=inp, out_channels=oup, dw_kernel_size=5, 
    stride=stride, act_fn=nn.ReLU, expand_ratio=t, se_ratio=None, **kwargs)
OPS['ir_7x7'] = lambda inp, oup, t, stride, kwargs: InvertedResidual(in_channels=inp, out_channels=oup, dw_kernel_size=7, 
    stride=stride, act_fn=nn.ReLU, expand_ratio=t, se_ratio=None, **kwargs)

# assign ops with given expand ratios
class OpWrapper:
    def __init__(self, t, op_func):
        self.t = t
        self.op_func = op_func
    def __call__(self, inp, oup, t, stride, kwargs):
        return self.op_func(inp, oup, self.t, stride, kwargs)

_t = [1, 3, 6]
new_ops = {}
for op in OPS:
    if 'ir' in op and 't' not in op:
        for given_t in _t:
            newop = op + f'_t{given_t}'
            func = OpWrapper(given_t, OPS[op])
            new_ops[newop] = func #lambda inp, oup, t, stride, kwargs: OPS[op](inp, oup, given_t, stride, kwargs)
for op in new_ops:
    OPS[op] = new_ops[op]

OPS['conv1x1'] = lambda inp, oup, t, stride, kwargs: ConvBnAct(in_channels=inp, out_channels=oup, kernel_size=1, stride=stride, **kwargs)
OPS['conv3x3'] = lambda inp, oup, t, stride, kwargs: ConvBnAct(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, **kwargs)
OPS['gavgp'] = lambda inp, oup, t, stride, kwargs: nn.AdaptiveAvgPool2d(1, **kwargs)
OPS['maxp'] = lambda inp, oup, t, stride, kwargs: nn.MaxPool2d(kernel_size=2, stride=stride, **kwargs)

OPS['linear_relu'] = lambda inp, oup, t, stride, kwargs: LinearReLU(inp, oup)

'''for NAS-Bench-Macro'''
OPS['ir_3x3_t3'] = lambda inp, oup, t, stride, kwargs: InvertedResidual(in_channels=inp, out_channels=oup, dw_kernel_size=3, 
    stride=stride, act_fn=nn.ReLU, expand_ratio=3, se_ratio=None, **kwargs)
OPS['ir_5x5_t6'] = lambda inp, oup, t, stride, kwargs: InvertedResidual(in_channels=inp, out_channels=oup, dw_kernel_size=5, 
    stride=stride, act_fn=nn.ReLU, expand_ratio=6, se_ratio=None, **kwargs)
OPS['ID'] = lambda inp, oup, t, stride, kwargs: Identity(in_channels=inp, out_channels=oup, kernel_size=1, stride=stride, **kwargs)


"""
==========================
basic operations & modules
==========================
"""

class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class LinearReLU(nn.Module):
    def __init__(self, inp, oup,):
        super(LinearReLU, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(inp, oup, bias=True),
            nn.ReLU(inplace=True))

    def forward(self, x):
        #if x.ndims != 2:
        if len(x.shape) != 2:
            x = x.view(x.shape[0], -1)
        return self.fc(x)


def conv2d(in_channels, out_channels, kernel_size, stride=1, pad_type='SAME', **kwargs):
    if pad_type == 'SAME' or pad_type == '':
        if isinstance(kernel_size, (tuple, list)):
            padding = [(kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2]
        else:
            padding = (kernel_size - 1) // 2
    elif pad_type == 'NONE':
        padding = 0
    else:
        raise NotImplementedError('Not supported padding type: {}.'.format(pad_type))
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, pad_type='SAME', act_fn=nn.ReLU, **attrs):
        super(ConvBnAct, self).__init__()
        for k, v in attrs.items():
            setattr(self, k, v)
        self.conv = conv2d(in_channels, out_channels, kernel_size, stride=stride, pad_type=pad_type, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = act_fn(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act(x)
        return x


class Identity(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, **kwargs):
        super(Identity, self).__init__()
        if in_channels != out_channels or stride != 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv = None

    def forward(self, x):
        if self.conv is not None:
            return self.conv(x)
        else:
            return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduce_channels, act_fn=nn.ReLU, gate_fn=nn.Sigmoid):
        super(SqueezeExcite, self).__init__()
        self.avgp = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_channels, reduce_channels, 1, bias=True)
        self.act_fn = act_fn(inplace=True)
        self.conv_expand = nn.Conv2d(reduce_channels, in_channels, 1, bias=True)
        self.gate_fn = gate_fn()

    def forward(self, x):
        x_se = self.avgp(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


"""
==========================
ShuffleNetV2-ops
==========================
"""
OPS['shuffle_3x3_se'] = lambda inp, oup, t, stride, kwargs: ShufflenetBlock(inp, oup, ksize=3, stride=stride, activation='HSwish', use_se=True)
OPS['shuffle_5x5_se'] = lambda inp, oup, t, stride, kwargs: ShufflenetBlock(inp, oup, ksize=5, stride=stride, activation='HSwish', use_se=True)
OPS['shuffle_7x7_se'] = lambda inp, oup, t, stride, kwargs: ShufflenetBlock(inp, oup, ksize=7, stride=stride, activation='HSwish', use_se=True)
OPS['shuffle_x_se'] = lambda inp, oup, t, stride, kwargs: ShufflenetBlock(inp, oup, ksize='x', stride=stride, activation='HSwish', use_se=True)


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


class ShufflenetBlock(nn.Module):

    def __init__(self, inp, oup, ksize, stride, activation='ReLU', use_se=False, **kwargs):
        super(ShufflenetBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7, 'x']
        base_mid_channels = oup // 2

        self.base_mid_channel = base_mid_channels
        self.ksize = ksize
        pad = ksize // 2 if ksize != 'x' else 3 // 2
        self.pad = pad
        if stride == 1:
            inp = inp // 2
            outputs = oup - inp
        else:
            outputs = oup // 2

        self.inp = inp


        if ksize != 'x':
            branch_main = [
                # pw
                nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(base_mid_channels),
                nn.ReLU(inplace=True) if activation == 'ReLU' else HSwish(inplace=True),
                # dw
                nn.Conv2d(base_mid_channels, base_mid_channels, ksize, stride, pad, groups=base_mid_channels, bias=False),
                nn.BatchNorm2d(base_mid_channels),
                # pw-linear
                nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True) if activation == 'ReLU' else HSwish(inplace=True),
            ]
        else:
            ksize = 3
            branch_main = [
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw
                nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(base_mid_channels),
                nn.ReLU(inplace=True) if activation == 'ReLU' else HSwish(inplace=True),
                # dw
                nn.Conv2d(base_mid_channels, base_mid_channels, 3, 1, 1, groups=base_mid_channels, bias=False),
                nn.BatchNorm2d(base_mid_channels),
                # pw
                nn.Conv2d(base_mid_channels, base_mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(base_mid_channels),
                nn.ReLU(inplace=True) if activation == 'ReLU' else HSwish(inplace=True),
                # dw
                nn.Conv2d(base_mid_channels, base_mid_channels, 3, 1, 1, groups=base_mid_channels, bias=False),
                nn.BatchNorm2d(base_mid_channels),
                # pw
                nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True) if activation == 'ReLU' else HSwish(inplace=True),
            ]
        if use_se:
            assert activation != 'ReLU'
            branch_main.append(SqueezeExcite(outputs, outputs // 4, act_fn=HSwish, gate_fn=HSigmoid))
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, outputs, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True) if activation == 'ReLU' else HSwish(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)



"""
==========================
DARTS-ops
==========================
"""
OPS['avg_pool_3x3'] = lambda inp, oup, t, stride, kwargs: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
OPS['max_pool_3x3'] = lambda inp, oup, t, stride, kwargs: nn.MaxPool2d(3, stride=stride, padding=1)
OPS['skip_connect'] = lambda inp, oup, t, stride, kwargs: nn.Identity() if stride == 1 else FactorizedReduce(inp, oup)
OPS['sep_conv_3x3'] = lambda inp, oup, t, stride, kwargs: SepConv(inp, oup, 3, stride)
OPS['sep_conv_5x5'] = lambda inp, oup, t, stride, kwargs: SepConv(inp, oup, 5, stride)
OPS['dil_conv_3x3'] = lambda inp, oup, t, stride, kwargs: DilConv(inp, oup, 3, stride, padding=2)
OPS['dil_conv_5x5'] = lambda inp, oup, t, stride, kwargs: DilConv(inp, oup, 5, stride, padding=4)


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride):
    super(ReLUConvBN, self).__init__()
    padding = (kernel_size - 1) // 2
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out)
    )

  def forward(self, x):
    return self.op(x)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride):
    super(SepConv, self).__init__()
    padding = (kernel_size - 1) // 2
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out),
      )

  def forward(self, x):
    return self.op(x)



class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=2):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out),
      )

  def forward(self, x):
    return self.op(x)


"""
==========================
blocks
==========================
"""

class InvertedResidualMixConv(nn.Module):
    '''Inverted Residual block from MobileNet V2'''
    def __init__(self, in_channels, out_channels, dw_kernel_size=3,
                 stride=1, pad_type='', act_fn=nn.ReLU, 
                 expand_ratio=1.0, se_ratio=0., se_gate_fn=nn.Sigmoid,
                 drop_connect_rate=0.0, use_residual=True, use_3x3_dw_only=False, **attrs):
        super(InvertedResidualMixConv, self).__init__()
        mid_channels = int(in_channels * expand_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = in_channels == out_channels and stride == 1 and use_residual
        self.drop_connect_rate = drop_connect_rate

        for k, v in attrs.items():
            # for edgenn: NAS and pruning
            setattr(self, k, v)

        # Point-wise convolution
        if expand_ratio == 1:
            self.conv_pw = nn.Sequential()
        else:
            self.conv_pw = nn.Sequential(
                conv2d(in_channels, mid_channels, 1, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                act_fn(inplace=True)
            )

        use_3x3_dw_only = False
        # Depth-wise convolution
        if not use_3x3_dw_only:
            self.conv_dw = nn.Sequential(
                #conv2d(mid_channels, mid_channels, dw_kernel_size, stride, groups=mid_channels, bias=False),
                MDConv(mid_channels, n_chunks=3, stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels),
                act_fn(inplace=True)
            )
        else:
            conv_dw = []
            for i in range((dw_kernel_size - 3) // 2 + 1):
                conv_dw.extend([
                    conv2d(mid_channels, mid_channels, 3, stride if i == 0 else 1, groups=mid_channels, bias=False),
                    nn.BatchNorm2d(mid_channels),
                ])
            conv_dw.append(act_fn(inplace=True))
            self.conv_dw = nn.Sequential(*conv_dw)

        # Squeeze-and-excitation
        if self.has_se:
            self.se = SqueezeExcite(
                mid_channels, reduce_channels=max(1, int(mid_channels * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)

        # Point-wise convolution
        self.conv_pw2 = nn.Sequential(
            conv2d(mid_channels, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        residual = x

        x = self.conv_pw(x)
        x = self.conv_dw(x)

        if self.has_se:
            x = self.se(x)

        x = self.conv_pw2(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_path(x, self.drop_connect_rate, self.training)
            x += residual

        return x




class InvertedResidual(nn.Module):
    '''Inverted Residual block from MobileNet V2'''
    def __init__(self, in_channels, out_channels, dw_kernel_size=3,
                 stride=1, pad_type='', act_fn=nn.ReLU, 
                 expand_ratio=1.0, se_ratio=0., se_gate_fn=nn.Sigmoid,
                 drop_connect_rate=0.0, use_residual=True, use_3x3_dw_only=False, **attrs):
        super(InvertedResidual, self).__init__()
        mid_channels = int(in_channels * expand_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = in_channels == out_channels and stride == 1 and use_residual
        self.drop_connect_rate = drop_connect_rate

        for k, v in attrs.items():
            # for edgenn: NAS and pruning
            setattr(self, k, v)

        # Point-wise convolution
        if expand_ratio == 1:
            self.conv_pw = nn.Sequential()
        else:
            self.conv_pw = nn.Sequential(
                conv2d(in_channels, mid_channels, 1, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                act_fn(inplace=True)
            )

        # Depth-wise convolution
        if not use_3x3_dw_only:
            self.conv_dw = nn.Sequential(
                conv2d(mid_channels, mid_channels, dw_kernel_size, stride, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                act_fn(inplace=True)
            )
        else:
            conv_dw = []
            for i in range((dw_kernel_size - 3) // 2 + 1):
                conv_dw.extend([
                    conv2d(mid_channels, mid_channels, 3, stride if i == 0 else 1, groups=mid_channels, bias=False),
                    nn.BatchNorm2d(mid_channels),
                ])
            conv_dw.append(act_fn(inplace=True))
            self.conv_dw = nn.Sequential(*conv_dw)

        # Squeeze-and-excitation
        if self.has_se:
            self.se = SqueezeExcite(
                mid_channels, reduce_channels=max(1, int(mid_channels * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)

        # Point-wise convolution
        self.conv_pw2 = nn.Sequential(
            conv2d(mid_channels, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        residual = x

        x = self.conv_pw(x)
        x = self.conv_dw(x)

        if self.has_se:
            x = self.se(x)

        x = self.conv_pw2(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_path(x, self.drop_connect_rate, self.training)
            x += residual

        return x


class DARTSCell(nn.Module):
    def __init__(self, cell_arch, c_prev_prev, c_prev, c, stride=1, reduction_prev=False, steps=4):
        super().__init__()
        self.cell_arch = cell_arch
        self.steps = steps
        self.preprocess0 = FactorizedReduce(c_prev_prev, c) if reduction_prev else \
                           ReLUConvBN(c_prev_prev, c, 1, stride=1)
        self.preprocess1 = ReLUConvBN(c_prev, c, 1, stride=1)

        if len(cell_arch[0]) != 0 and isinstance(cell_arch[0][0], str):
            # DARTS-like genotype, convert it to topo-free type
            cell_arch = [[cell_arch[idx*2], cell_arch[idx*2+1]] for idx in range(len(cell_arch) // 2)]

        self.ops = nn.ModuleList()
        self.inputs = []
        for step in cell_arch:
            step_ops = nn.ModuleList()
            step_inputs = []
            for op_name, input_idx in step:
                step_ops += [OPS[op_name](c, c, None, stride if input_idx < 2 else 1, {})]
                step_inputs.append(input_idx)
            self.ops += [step_ops]
            self.inputs.append(step_inputs)

    def forward(self, s0, s1, drop_path_rate=0.):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        for step_idx, (step_inputs, step_ops) in enumerate(zip(self.inputs, self.ops)):
            step_outs = []
            for input_idx, op in zip(step_inputs, step_ops):
                out = op(states[input_idx])
                if drop_path_rate > 0. and not isinstance(op, (FactorizedReduce, nn.Identity)):
                    out = drop_path(out, drop_path_rate, self.training)
                step_outs.append(out)
            states.append(sum(step_outs))

        return torch.cat(states[-4:], dim=1)


"""
=========================
Auxiliary Heads
=========================
"""
class AuxiliaryHead(nn.Module):

    def __init__(self, C, num_classes, avg_pool_stride=2):
        """with avg_pol_stride=2, assuming input size 14x14"""
        super(AuxiliaryHead, self).__init__()
        self.features = nn.Sequential(
          nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
          nn.Conv2d(C, 128, 1, bias=False),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.Conv2d(128, 768, 2, bias=False),
          nn.BatchNorm2d(768),
          nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x



