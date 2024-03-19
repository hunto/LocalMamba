import torch
import torch.nn as nn
import torch.distributed as dist
import logging


logger = logging.getLogger()


def recal_bn(model, train_loader, recal_bn_iters=200, module=None):
    status = model.training
    model.eval()
    m = model if module is None else module
    if recal_bn_iters > 0:
        # recal bn
        logger.info(f'recalculating bn stats {recal_bn_iters} iters')
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d) or issubclass(mod.__class__, nn.BatchNorm2d):
                mod.reset_running_stats()
                # for small recal_bn_iters like 20, must set mod.momentum = None
                # for big recal_bn_iters like 300, mod.momentum can be 0.1
                mod.momentum = None
                mod.train()

        with torch.no_grad():
            cnt = 0
            while cnt < recal_bn_iters:
                for i, (images, target) in enumerate(train_loader):
                    images = images.cuda()
                    target = target.cuda()
                    output = model(images)
                    cnt += 1
                    if i % 20 == 0 or cnt == recal_bn_iters:
                        logger.info(f'recal bn iter {i}')
                    if cnt >= recal_bn_iters:
                        break

    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d) or issubclass(mod.__class__, nn.BatchNorm2d):
            if mod.track_running_stats:
                dist.all_reduce(mod.running_mean)
                dist.all_reduce(mod.running_var)
                mod.running_mean /= dist.get_world_size()
                mod.running_var /= dist.get_world_size()
                mod.momentum = 0.1
    model.train(status)
