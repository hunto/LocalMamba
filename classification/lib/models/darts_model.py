import torch
import torch.nn as nn
from .operations import DARTSCell, AuxiliaryHead


def gen_darts_model(net_cfg, dataset='imagenet', drop_rate=0., drop_path_rate=0., auxiliary_head=False, **kwargs):
    if dataset.lower() == 'imagenet':
        dataset = 'imagenet'
    elif dataset.lower() in ['cifar', 'cifar10', 'cifar100']:
        dataset = 'cifar'
    model = DARTSModel(net_cfg, dataset, drop_rate, drop_path_rate, auxiliary_head=auxiliary_head)
    return model


class DARTSModel(nn.Module):
    def __init__(self, net_cfg, dataset='imagenet', drop_rate=0., drop_path_rate=0., auxiliary_head=False):
        super(DARTSModel, self).__init__()
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        cell_normal = eval(net_cfg['genotype']['normal'])
        cell_reduce = eval(net_cfg['genotype']['reduce'])
        init_channels = net_cfg.get('init_channels', 48)
        layers = net_cfg.get('layers', 14)
        cell_multiplier = net_cfg.get('cell_multiplier', 4)
        num_classes = net_cfg.get('num_classes', 1000)
        
        reduction_layers = [layers // 3, layers * 2 // 3]
        C = init_channels

        if dataset == 'imagenet':
            C_curr = C
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, C_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C_curr // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C_curr // 2, C_curr, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
    
            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(C_curr, C_curr, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
        elif dataset == 'cifar':
            stem_multiplier = 3
            C_curr = C * stem_multiplier
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, C_curr, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
            self.stem1 = nn.Identity()

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        # cell blocks
        self.nas_cells = nn.Sequential()
        reduction_prev = dataset == 'imagenet'
        for layer_idx in range(layers):
            s = 1
            cell_arch = cell_normal
            if layer_idx in reduction_layers:
                s = 2
                C_curr *= 2
                cell_arch = cell_reduce
            cell = DARTSCell(cell_arch, C_prev_prev, C_prev, C_curr, stride=s, reduction_prev=reduction_prev)
            self.nas_cells.add_module('cell_{}'.format(layer_idx), cell)
            reduction_prev = (s == 2)
            C_prev_prev, C_prev = C_prev, C_curr * cell_multiplier                                        
            if auxiliary_head and layer_idx == 2 * layers // 3:
                C_to_auxiliary = C_prev
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        if auxiliary_head:
            object.__setattr__(self, 'module_to_auxiliary', cell)
            self.auxiliary_head = nn.Sequential(
                nn.ReLU(inplace=False),
                AuxiliaryHead(C_to_auxiliary, num_classes, avg_pool_stride=2 if dataset=='imagenet' else 3)
            )

    def get_classifier(self):
        return self.classifier

    def forward(self, x):
        s0 = self.stem0(x)
        s1 = self.stem1(s0)
        for cell in self.nas_cells:
            s0, s1 = s1, cell(s0, s1, self.drop_path_rate)
        x = self.pool(s1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

