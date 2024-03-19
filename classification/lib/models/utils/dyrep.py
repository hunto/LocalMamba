import random
import logging
import warnings
import torch
import torch.distributed as dist
import torch.nn as nn

from lib.utils.optim import get_params
from lib.utils.misc import AverageMeter
from .dbb.dbb_block import DiverseBranchBlock


logger = logging.getLogger()


class DyRep(object):
    def __init__(self,
                 model,
                 optimizer,
                 recal_bn_fn=None,
                 grow_metric='synflow',
                 dbb_branches=[1, 1, 1, 1, 1, 1, 1],
                 filter_bias_and_bn=False):
        self.model = model
        self.recal_bn_fn = recal_bn_fn
        self.optimizer = optimizer
        self.filter_bias_and_bn = filter_bias_and_bn  # used in optimizer get_params

        accept_metrics = ('grad_norm', 'snip', 'synflow', 'random')
        assert grow_metric in accept_metrics, \
            f'DyRep supports metrics {accept_metrics}, ' \
            f'but gets {grow_metric}'
        self.grow_metric = grow_metric
        self.dbb_branches = dbb_branches
        # valid dbb branches for conv with unequal shapes of input and output
        self.dbb_branches_unequal = [
            v if i not in (0, 4, 5, 6) else 0
            for i, v in enumerate(dbb_branches)
        ]

        # dict for recording the metric of each conv modules
        self._metric_records = {}
        self._weight_records = {}

        self.new_param_group = None

        self.last_growed_module = 'none'

    def _get_module(self, path):
        path_split = path.split('.')
        m = self.model
        for key in path_split:
            if not hasattr(m, key):
                return None
            m = getattr(m, key)
        return m

    def record_metrics(self):
        for k, m in self.model.named_modules():
            if not isinstance(m, nn.Conv2d) \
                    or m.kernel_size[0] != m.kernel_size[1] \
                    or m.kernel_size[0] == 1 \
                    or k.count('dbb') >= 2:
                # Requirements for growing the module:
                # 1. the module is a nn.Conv2d module;
                # 2. it must has the same kernel_size (>1) in `h` and `w` axes;
                # 3. we restrict the number of growths in each layer.
                continue

            if m.weight.grad is None:
                continue
            grad = m.weight.grad.data.reshape(-1)
            weight = m.weight.data.reshape(-1)

            if self.grow_metric == 'grad_norm':
                metric_val = grad.norm().item()
            elif self.grow_metric == 'snip':
                metric_val = (grad * weight).abs().sum().item()
            elif self.grow_metric == 'synflow':
                metric_val = (grad * weight).sum().item()
            elif self.grow_metric == 'random':
                metric_val = random.random()

            if k not in self._metric_records:
                self._metric_records[k] = AverageMeter(dist=True)
            self._metric_records[k].update(metric_val)

    def _grow(self, metric_records_sorted, topk=1):
        if len(metric_records_sorted) == 0:
            return
        for i in range(topk):
            conv_to_grow = metric_records_sorted[i][0]
            logger.info('grow: {}'.format(conv_to_grow))
            len_parent_str = conv_to_grow.rfind('.')
            if len_parent_str != -1:
                parent = conv_to_grow[:len_parent_str]
                conv_key = conv_to_grow[len_parent_str + 1:]
                # get the target conv module and its parent
                parent_m = self._get_module(parent)
            else:
                conv_key = conv_to_grow
                parent_m = self.model
            conv_m = getattr(parent_m, conv_key, None)
            # replace target conv module with DBB
            conv_m_padding = conv_m.padding[0]
            conv_m_kernel_size = conv_m.kernel_size[0]

            if conv_m_padding == conv_m_kernel_size // 2:
                dbb_branches = self.dbb_branches.copy()
            else:
                dbb_branches = self.dbb_branches_unequal.copy()
            dbb_block = DiverseBranchBlock(
                conv_m.in_channels,
                conv_m.out_channels,
                conv_m_kernel_size,
                stride=conv_m.stride,
                groups=conv_m.groups,
                padding=conv_m_padding,
                ori_conv=conv_m,
                branches=dbb_branches,
                use_bn=True,
                bn=nn.BatchNorm2d,
                recal_bn_fn=self.recal_bn_fn).cuda()
            setattr(parent_m, conv_key, dbb_block)
            dbb_block._reset_dbb(conv_m.weight, conv_m.bias)
            self.last_growed_module = conv_to_grow
        logger.info(str(self.model))

    def _cut(self, dbb_key, cut_branches, remove_bn=False):
        dbb_m = self._get_module(dbb_key)
        if dbb_m is None:
            return
        if sum(cut_branches) == 1:
            # only keep the original 3x3 conv
            parent = self._get_module(dbb_key[:dbb_key.rfind('.')])
            weight, bias = dbb_m.get_actual_kernel()
            conv = nn.Conv2d(dbb_m.in_channels,
                             dbb_m.out_channels,
                             dbb_m.kernel_size,
                             stride=dbb_m.stride,
                             groups=dbb_m.groups,
                             padding=dbb_m.padding,
                             bias=True).cuda()
            conv.weight.data = weight
            conv.bias.data = bias
            setattr(parent, dbb_key[dbb_key.rfind('.') + 1:], conv)
        else:
            dbb_m.cut_branch(cut_branches)

    def _reset_optimizer(self):
        param_groups = get_params(self.model, lr=0.1, weight_decay=1e-5, filter_bias_and_bn=self.filter_bias_and_bn, sort_params=True)

        # remove the states of removed paramters
        assert len(param_groups) == len(self.optimizer.param_groups)
        for param_group, param_group_old in zip(param_groups, self.optimizer.param_groups):
            params, params_old = param_group['params'], param_group_old['params']
            params = set(params)
            for param_old in params_old:
                if param_old not in params:
                    if param_old in self.optimizer.state:
                        del self.optimizer.state[param_old]
            param_group_old['params'] = param_group['params']

    def adjust_model(self):
        records = {}
        for key in self._metric_records:
            records[key] = self._metric_records[key].avg

        metric_records_sorted = sorted(records.items(),
                                       key=lambda item: item[1],
                                       reverse=True)

        logger.info('metric: {}'.format(metric_records_sorted))
        self._grow(metric_records_sorted)

        # reset records
        self._metric_records = {}

        for k, m in self.model.named_modules():
            if isinstance(m, DiverseBranchBlock):
                weights = m.branch_weights()
                logger.info(k + ': ' + str(weights))
                valid_weights = torch.tensor(
                    [x for x in weights[:3] + weights[4:] if x not in [-1, 1]])
                if valid_weights.std() > 0.02:
                    mean = valid_weights.mean()
                    # cut those branches less than 0.1
                    need_cut = False
                    cut_branches = [1] * len(weights)
                    for idx in range(len(weights)):
                        if weights[idx] < mean and weights[idx] < 0.1:
                            cut_branches[idx] = 0
                            if weights[idx] != -1:
                                need_cut = True
                    if need_cut:
                        self._cut(k, cut_branches)
                        logger.info(
                            f'cut: {k}, new branches: {cut_branches}')

        self._reset_optimizer()

    def state_dict(self):
        # save dbb graph
        res = {}
        res['dbb_graph'] = self.dbb_graph()
        return res

    def load_state_dict(self, state_dict):
        if 'dbb_graph' in state_dict:
            self.load_dbb_graph(state_dict['dbb_graph'])


    def dbb_graph(self):
        dbb_list = []

        def traverse(parent, prefix=''):
            for k, m in parent.named_children():
                path = prefix + '.' + k if prefix != '' else k
                if isinstance(m, DiverseBranchBlock):
                    dbb_list.append((path, m.branches))
                traverse(m, prefix=path)

        traverse(self.model)
        print(dbb_list)
        return dbb_list

    def load_dbb_graph(self, dbb_list: list):
        if dbb_list is None or len(dbb_list) == 0:
            return
        print(dbb_list)
        assert not any(
            [isinstance(m, DiverseBranchBlock)
             for m in self.model.modules()]), 'model must be clean'
        for key, branches in dbb_list:
            parent = self._get_module(key[:key.rfind('.')])
            conv_key = key[key.rfind('.') + 1:]
            conv_m = getattr(parent, conv_key)
            dbb_m = DiverseBranchBlock(conv_m.in_channels,
                                       conv_m.out_channels,
                                       conv_m.kernel_size[0],
                                       stride=conv_m.stride,
                                       groups=conv_m.groups,
                                       padding=conv_m.padding[0],
                                       ori_conv=conv_m,
                                       branches=branches,
                                       use_bn=True)
            setattr(parent, conv_key, dbb_m)
        self.model.cuda()
        # reset optimizer
        if self.optimizer is not None:
            self._reset_optimizer()
        # print(self.model)
