import torch.nn as nn

from .dbb.dbb_block import DiverseBranchBlock


# Convert all the 3x3 convs in the model to DBB blocks
def convert_to_dbb(model, ignore_key=None, dbb_branches=[1, 1, 1, 1, 0, 0, 0]):
    named_children = list(model.named_children())
    next_bn = False
    for k, m in named_children:
        if k == '':
            continue
        if ignore_key is not None and k.startswith(ignore_key):
            continue
        if isinstance(
                m, nn.Conv2d
        ) and m.kernel_size[0] == 3 and m.kernel_size[0] == m.kernel_size[1]:
            # dbb_branches = [1, 1, 1, 1, 1, 1, 0]
            # dbb_branches = [1, 1, 1, 1, 0, 0, 0]
            if m.padding[0] != m.kernel_size[0] // 2:
                dbb_branches_ = [0, 1, 1, 1, 0, 0, 0]
            else:
                dbb_branches_ = dbb_branches
            setattr(
                model, k,
                DiverseBranchBlock(m.in_channels,
                                   m.out_channels,
                                   m.kernel_size[0],
                                   stride=m.stride,
                                   groups=m.groups,
                                   padding=m.padding[0],
                                   ori_conv=None,
                                   branches=dbb_branches_,
                                   use_bn=True))
            next_bn = True
        if isinstance(m, nn.BatchNorm2d) and next_bn:
            setattr(model, k, nn.Identity())
            next_bn = False
        else:
            convert_to_dbb(m, ignore_key=None)
    print(model)
