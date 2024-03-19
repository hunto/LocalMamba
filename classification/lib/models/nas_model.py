import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import OPS, AuxiliaryHead


def _initialize_weight_goog(m):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # fan-out
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(0)  # fan-out
        init_range = 1.0 / math.sqrt(n)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def _initialize_weight_default(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')


class NASModel(nn.Module):
    def __init__(self, net_cfg, weight_init='goog', drop_rate=0.2, drop_path_rate=0.0, auxiliary_head=False, **kwargs):
        super(NASModel, self).__init__()
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        if self.drop_path_rate != 0.:
            raise NotImplementedError('Drop path is not implemented in NAS model.')

        backbone_cfg = net_cfg.pop('backbone')
        self.features = nn.Sequential()
        downsample_num = 0
        for layer in backbone_cfg:
            if len(backbone_cfg[layer]) == 5:
                stride, inp, oup, t, op = backbone_cfg[layer]
                n = 1
                kwargs = {}
            elif len(backbone_cfg[layer]) == 6 and isinstance(backbone_cfg[layer][-1], dict):
                stride, inp, oup, t, op, kwargs = backbone_cfg[layer]
                n = 1
            elif len(backbone_cfg[layer]) == 6:
                n, stride, inp, oup, t, op = backbone_cfg[layer]
                kwargs = {}
            elif len(backbone_cfg[layer]) == 7:
                n, stride, inp, oup, t, op, kwargs = backbone_cfg[layer]
            else:
                raise RuntimeError(f'Invalid layer configuration: {backbone_cfg[layer]}')

            for idx in range(n):
                layer_ = layer + f'_{idx}' if n > 1 else layer
                if isinstance(t, (list, tuple)) or isinstance(op, (list, tuple)):
                    # NAS supernet
                    if not isinstance(t, (list, tuple)):
                        t = [t]
                    if not isinstance(op, (list, tuple)):
                        op = [op]
                    from edgenn.models import ListChoice
                    blocks = []
                    for t_ in t:
                        for op_ in op:
                            if op_ == 'id':
                                # add it later
                                continue
                            blocks.append(OPS[op_](inp, oup, t_, stride, kwargs))
                    if 'id' in op:
                        blocks.append(OPS['id'](inp, oup, 1, stride, kwargs))
                    self.features.add_module(layer_, ListChoice(blocks))
                else:
                    if t is None:
                        t = 1
                    self.features.add_module(layer_, OPS[op](inp, oup, t, stride, kwargs))
                    if stride == 2:
                        downsample_num += 1
                        if auxiliary_head and downsample_num == 5:
                            # auxiliary head added after the 5-th downsampling layer
                            object.__setattr__(self, 'module_to_auxiliary', self.features[-1])
                            C_to_auxiliary = oup
                inp = oup
                stride = 1
        
        # build head
        head_cfg = net_cfg.pop('head')
        self.classifier = nn.Sequential()
        for layer in head_cfg:
            self.classifier.add_module(layer, nn.Linear(head_cfg[layer]['dim_in'], head_cfg[layer]['dim_out']))

        if auxiliary_head:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, 1000)
    
        # init weight
        for m in self.modules():
            if weight_init == 'goog':
                _initialize_weight_goog(m)
            else:
                _initialize_weight_default(m)

    def get_classifier(self):
        return self.classifier

    def forward(self, x):
        x = self.features(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def gen_nas_model(net_cfg, drop_rate=0.2, drop_path_rate=0.0, auxiliary_head=False, **kwargs):
    model = NASModel(
        net_cfg,      
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        auxiliary_head=auxiliary_head
    )
    return model

