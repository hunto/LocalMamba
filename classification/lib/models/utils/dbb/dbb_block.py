import math

import torch
import torch.nn as nn

from .dbb_transforms import (transI_fusebn, transII_addbranch,
                             transIII_1x1_kxk, transVI_multiscale,
                             transV_avg, transIX_bn_to_1x1)


class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 bn=nn.BatchNorm2d):
        super(ConvBN, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)

        self.bn = bn(num_features=out_channels,
                     affine=True,
                     track_running_stats=True)
        self.deployed = False

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output


class BNAndPad(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 last_conv_bias=None,
                 bn=nn.BatchNorm2d):
        super(BNAndPad, self).__init__()
        self.bn = bn(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels
        self.last_conv_bias = last_conv_bias

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            bias = -self.bn.running_mean
            if self.last_conv_bias is not None:
                bias += self.last_conv_bias
            pad_values = self.bn.bias.data + self.bn.weight.data * (
                bias / torch.sqrt(self.bn.running_var + self.bn.eps))
            ''' pad '''
            n, c, h, w = output.size()
            values = pad_values.view(1, -1, 1, 1)
            w_values = values.expand(n, -1, self.pad_pixels, w)
            x = torch.cat([w_values, output, w_values], dim=2)
            h = h + self.pad_pixels * 2
            h_values = values.expand(n, -1, h, self.pad_pixels)
            x = torch.cat([h_values, x, h_values], dim=3)
            output = x
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class DiverseBranchBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            branches=[1, 1, 1, 1, 1, 1, 1
                      ],  # stands for 1x1, 1x1_kxk, 1x1_avg, kxk, 1xk, kx1, id
            internal_channels=None,  # internal channel between 1x1 and kxk
            nonlinear=None,
            ori_conv=None,
            padding=None,
            bn=nn.BatchNorm2d,
            recal_bn_fn=None,
            **kwargs):
        super(DiverseBranchBlock, self).__init__()
        if isinstance(stride, tuple):
            stride = stride[0]
        if not (out_channels == in_channels and stride == 1):
            branches[6] = 0
        assert branches[3] == 1  # original kxk branch should always be active
        self.deployed = False
        self.branches = branches
        if nonlinear is None:
            self.nonlinear = nn.Sequential()
        else:
            self.nonlinear = nonlinear
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

        self.active_branch_num = sum(branches)
        if branches[0]:
            self.dbb_1x1 = ConvBN(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=stride,
                                  padding=0,
                                  groups=groups,
                                  bn=bn)
        if branches[1]:
            if internal_channels is None:
                internal_channels = in_channels
            self.dbb_1x1_kxk = nn.Sequential()
            self.dbb_1x1_kxk.add_module(
                'conv1',
                nn.Conv2d(in_channels=in_channels,
                          out_channels=internal_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          groups=groups,
                          bias=False))
            self.dbb_1x1_kxk.add_module(
                'bn1',
                BNAndPad(pad_pixels=padding,
                         num_features=internal_channels,
                         affine=True,
                         last_conv_bias=self.dbb_1x1_kxk.conv1.bias,
                         bn=bn))
            self.dbb_1x1_kxk.add_module(
                'conv2',
                nn.Conv2d(in_channels=internal_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=0,
                          groups=groups,
                          bias=False))
            self.dbb_1x1_kxk.add_module(
                'bn2',
                bn(num_features=out_channels,
                   affine=True,
                   track_running_stats=True))
        if branches[2]:
            self.dbb_1x1_avg = nn.Sequential()
            if self.groups < self.out_channels:
                self.dbb_1x1_avg.add_module(
                    'conv',
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              groups=groups,
                              bias=False))
                self.dbb_1x1_avg.add_module(
                    'bn',
                    BNAndPad(pad_pixels=padding,
                             num_features=out_channels,
                             last_conv_bias=self.dbb_1x1_avg.conv.bias,
                             bn=bn))
                self.dbb_1x1_avg.add_module(
                    'avg',
                    nn.AvgPool2d(kernel_size=kernel_size,
                                 stride=stride,
                                 padding=0))
            else:
                self.dbb_1x1_avg.add_module(
                    'avg',
                    nn.AvgPool2d(kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding))
            self.dbb_1x1_avg.add_module(
                'avgbn',
                bn(
                    num_features=out_channels,
                    affine=True,
                    track_running_stats=True,
                ))
        if branches[3]:
            self.dbb_kxk = ConvBN(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=True,
                                  bn=bn)
        if branches[4]:
            self.dbb_1xk = ConvBN(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(1, kernel_size),
                                  stride=stride,
                                  padding=(0, self.padding),
                                  dilation=dilation,
                                  groups=groups,
                                  bias=False,
                                  bn=bn)
        if branches[5]:
            self.dbb_kx1 = ConvBN(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(kernel_size, 1),
                                  stride=stride,
                                  padding=(self.padding, 0),
                                  dilation=dilation,
                                  groups=groups,
                                  bias=False,
                                  bn=bn)
        if branches[6]:
            self.dbb_id = bn(
                num_features=out_channels,
                affine=True,
                track_running_stats=True,
            )

        if ori_conv is not None:
            self.recal_bn_fn = recal_bn_fn

    def branch_weights(self):
        def _cal_weight(data):
            return data.abs().mean().item()  # L1

        weights = [-1] * len(self.branches)
        kxk_weight = _cal_weight(self.dbb_kxk.bn.weight.data)
        # Make the weight of kxk branch as 1,
        # this is for better generalization of the thrd value (lambda)
        weights[3] = 1 
        if self.branches[0]:
            weights[0] = _cal_weight(self.dbb_1x1.bn.weight.data) / kxk_weight
        if self.branches[1]:
            weights[1] = _cal_weight(
                self.dbb_1x1_kxk[-1].weight.data) / kxk_weight
        if self.branches[2]:
            weights[2] = _cal_weight(
                self.dbb_1x1_avg[-1].weight.data) / kxk_weight
        if self.branches[4]:
            weights[4] = _cal_weight(self.dbb_1xk.bn.weight.data) / kxk_weight
        if self.branches[5]:
            weights[5] = _cal_weight(self.dbb_kx1.bn.weight.data) / kxk_weight
        if self.branches[6]:
            weights[6] = _cal_weight(self.dbb_id.weight.data) / kxk_weight
        return weights

    def _reset_dbb(self,
                   kernel,
                   bias,
                   no_init_branches=[0, 0, 0, 0, 0, 0, 0, 0]):
        self._init_branch(self.dbb_kxk, set_zero=True, norm=1)
        if self.branches[0] and no_init_branches[0] == 0:
            self._init_branch(self.dbb_1x1)
        if self.branches[1] and no_init_branches[1] == 0:
            self._init_branch(self.dbb_1x1_kxk)
        if self.branches[2] and no_init_branches[2] == 0:
            self._init_branch(self.dbb_1x1_avg)
        if self.branches[4] and no_init_branches[4] == 0:
            self._init_branch(self.dbb_1xk)
        if self.branches[5] and no_init_branches[5] == 0:
            self._init_branch(self.dbb_kx1)
        if self.branches[6] and no_init_branches[6] == 0:
            self._init_branch(self.dbb_id)

        if self.recal_bn_fn is not None and sum(
                no_init_branches) == 0 and isinstance(kernel, nn.Parameter):
            self.dbb_kxk.conv.weight.data.copy_(kernel)
            if bias is not None:
                self.dbb_kxk.conv.bias = bias
            self.recal_bn_fn(self)
            self.dbb_kxk.bn.reset_running_stats()
        cur_w, cur_b = self.get_actual_kernel(ignore_kxk=True)
        # reverse dbb transform
        new_w = kernel.data.to(cur_w.device) - cur_w
        if bias is not None:
            new_b = bias.data.to(cur_b.device) - cur_b
        else:
            new_b = -cur_b

        if isinstance(self.dbb_kxk.conv, nn.Conv2d):
            if isinstance(self.dbb_kxk.bn, nn.BatchNorm2d):
                self.dbb_kxk.bn.weight.data.fill_(1.)
                self.dbb_kxk.bn.bias.data.zero_()
            self.dbb_kxk.conv.weight.data = new_w
            self.dbb_kxk.conv.bias.data = new_b
        elif isinstance(self.dbb_kxk.conv, DiverseBranchBlock):
            self.dbb_kxk.conv._reset_dbb(new_w, new_b)

    def _init_branch(self, branch, set_zero=False, norm=0.01):
        bns = []
        for m in branch.modules():
            if isinstance(m, nn.Conv2d):
                if set_zero:
                    m.weight.data.zero_()
                else:
                    n = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels  # fan-out
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                bns.append(m)
        for idx, m in enumerate(bns):
            m.reset_parameters()
            m.reset_running_stats()
            if idx == len(bns) - 1:
                m.weight.data.fill_(norm)  # set to a small value
            else:
                m.weight.data.fill_(1.)
            if m.bias is not None:
                m.bias.data.zero_()

    def get_actual_kernel(self, ignore_kxk=False):
        if self.deployed:
            return self.conv_deployed.weight.data, self.conv_deployed.bias.data
        ws = []
        bs = []
        if not ignore_kxk:  # kxk-bn
            if isinstance(self.dbb_kxk.conv, nn.Conv2d):
                w, b = self.dbb_kxk.conv.weight, self.dbb_kxk.conv.bias
            elif isinstance(self.dbb_kxk.conv, DiverseBranchBlock):
                w, b = self.dbb_kxk.conv.get_actual_kernel()
            if not isinstance(self.dbb_kxk.bn, nn.Identity):
                w, b = transI_fusebn(w, self.dbb_kxk.bn, b)
            ws.append(w.unsqueeze(0))
            bs.append(b.unsqueeze(0))
        if self.branches[0]:  # 1x1-bn
            w_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight,
                                         self.dbb_1x1.bn,
                                         self.dbb_1x1.conv.bias)
            w_1x1 = transVI_multiscale(w_1x1, self.kernel_size)
            ws.append(w_1x1.unsqueeze(0))
            bs.append(b_1x1.unsqueeze(0))
        if self.branches[1]:  # 1x1-bn-kxk-bn
            if isinstance(self.dbb_1x1_kxk.conv2, nn.Conv2d):
                w_1x1_kxk, b_1x1_kxk = transI_fusebn(
                    self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2,
                    self.dbb_1x1_kxk.conv2.bias)
            elif isinstance(self.dbb_1x1_kxk.conv2, DiverseBranchBlock):
                w_1x1_kxk, b_1x1_kxk = \
                    self.dbb_1x1_kxk.conv2.get_actual_kernel()
                w_1x1_kxk, b_1x1_kxk = transI_fusebn(w_1x1_kxk,
                                                     self.dbb_1x1_kxk.bn2,
                                                     b_1x1_kxk)
            w_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(
                self.dbb_1x1_kxk.conv1.weight, self.dbb_1x1_kxk.bn1,
                self.dbb_1x1_kxk.conv1.bias)
            w_1x1_kxk, b_1x1_kxk = transIII_1x1_kxk(w_1x1_kxk_first,
                                                    b_1x1_kxk_first,
                                                    w_1x1_kxk,
                                                    b_1x1_kxk,
                                                    groups=self.groups)
            ws.append(w_1x1_kxk.unsqueeze(0))
            bs.append(b_1x1_kxk.unsqueeze(0))
        if self.branches[2]:  # 1x1-bn-avg-bn
            w_1x1_avg = transV_avg(self.out_channels, self.kernel_size,
                                   self.groups)
            w_1x1_avg, b_1x1_avg = transI_fusebn(
                w_1x1_avg.to(self.dbb_1x1_avg.avgbn.weight.device),
                self.dbb_1x1_avg.avgbn, None)
            if self.groups < self.out_channels:
                w_1x1_avg_first, b_1x1_avg_first = transI_fusebn(
                    self.dbb_1x1_avg.conv.weight, self.dbb_1x1_avg.bn,
                    self.dbb_1x1_avg.conv.bias)
                w_1x1_avg, b_1x1_avg = transIII_1x1_kxk(w_1x1_avg_first,
                                                        b_1x1_avg_first,
                                                        w_1x1_avg,
                                                        b_1x1_avg,
                                                        groups=self.groups)
            ws.append(w_1x1_avg.unsqueeze(0))
            bs.append(b_1x1_avg.unsqueeze(0))
        if self.branches[4]:  # 1xk-bn
            w_1xk, b_1xk = transI_fusebn(self.dbb_1xk.conv.weight,
                                         self.dbb_1xk.bn,
                                         self.dbb_1xk.conv.bias)
            w_1xk = transVI_multiscale(w_1xk, self.kernel_size)
            ws.append(w_1xk.unsqueeze(0))
            bs.append(b_1xk.unsqueeze(0))
        if self.branches[5]:  # kx1-bn
            w_kx1, b_kx1 = transI_fusebn(self.dbb_kx1.conv.weight,
                                         self.dbb_kx1.bn,
                                         self.dbb_kx1.conv.bias)
            w_kx1 = transVI_multiscale(w_kx1, self.kernel_size)
            ws.append(w_kx1.unsqueeze(0))
            bs.append(b_kx1.unsqueeze(0))
        if self.branches[6]:  # BN
            w_id, b_id = transIX_bn_to_1x1(self.dbb_id,
                                           self.dbb_kxk.conv.in_channels,
                                           self.dbb_kxk.conv.groups)
            w_id = transVI_multiscale(w_id, self.kernel_size)
            ws.append(w_id.unsqueeze(0))
            bs.append(b_id.unsqueeze(0))

        ws = torch.cat(ws)
        bs = torch.cat(bs)

        return transII_addbranch(ws, bs)

    def switch_to_deploy(self):
        if self.deployed:
            return
        w, b = self.get_actual_kernel()

        self.conv_deployed = nn.Conv2d(in_channels=self.in_channels,
                                       out_channels=self.out_channels,
                                       kernel_size=self.kernel_size,
                                       stride=self.stride,
                                       padding=self.padding,
                                       dilation=self.dilation,
                                       groups=self.groups,
                                       bias=True)

        self.conv_deployed.weight.data = w
        self.conv_deployed.bias.data = b
        for para in self.parameters():
            para.detach_()
        if self.branches[0]:
            self.__delattr__('dbb_1x1')
        if self.branches[1]:
            self.__delattr__('dbb_1x1_kxk')
        if self.branches[2]:
            self.__delattr__('dbb_1x1_avg')
        if self.branches[3]:
            self.__delattr__('dbb_kxk')
        if self.branches[4]:
            self.__delattr__('dbb_1xk')
        if self.branches[5]:
            self.__delattr__('dbb_kx1')
        if self.branches[6]:
            self.__delattr__('dbb_id')
        self.deployed = True

    def forward(self, inputs):
        if self.deployed:
            return self.nonlinear(self.conv_deployed(inputs))

        branch_outs = []
        branch_outs.append(self.dbb_kxk(inputs))
        if self.branches[0]:
            branch_outs.append(self.dbb_1x1(inputs))
        if self.branches[1]:
            branch_outs.append(self.dbb_1x1_kxk(inputs))
        if self.branches[2]:
            branch_outs.append(self.dbb_1x1_avg(inputs))
        if self.branches[4]:
            branch_outs.append(self.dbb_1xk(inputs))
        if self.branches[5]:
            branch_outs.append(self.dbb_kx1(inputs))
        if self.branches[6]:
            branch_outs.append(self.dbb_id(inputs))

        out = self.nonlinear(torch.stack(branch_outs).sum(0))
        return out

    def cut_branch(self, branches):
        ori_w, ori_b = self.get_actual_kernel()
        _branch_names = [
            'dbb_1x1', 'dbb_1x1_kxk', 'dbb_1x1_avg', 'dbb_kxk', 'dbb_1xk',
            'dbb_kx1', 'dbb_id'
        ]
        for idx, status in enumerate(branches):
            if status == 0 and self.branches[idx] == 1:
                self.branches[idx] = 0
                self.__delattr__(_branch_names[idx])
        self._reset_dbb(ori_w, ori_b, no_init_branches=branches)

