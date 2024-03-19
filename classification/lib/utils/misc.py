import shutil
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import logging
logger = logging.getLogger()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.item() / batch_size)
    return res


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, dist=False):
        self.dist = dist
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_val = 0
        self.local_sum = 0
        self.local_count = 0

    def update(self, val, n=1):
        self.local_val = val
        self.local_sum += val * n
        self.local_count += n
        if not self.dist:
            self.val = self.local_val
            self.sum = self.local_sum
            self.count = self.local_count
            self.avg = self.sum / self.count
        else:
            self._dist_reduce()

    def _dist_reduce(self):
        '''gather results from all ranks'''
        reduce_tensor = torch.Tensor([self.local_val, self.local_sum, self.local_count]).cuda()
        dist.all_reduce(reduce_tensor)
        world_size = dist.get_world_size()
        self.val = reduce_tensor[0].item() / world_size
        self.sum = reduce_tensor[1].item()
        self.count = reduce_tensor[2].item()
        self.avg = self.sum / self.count


class CheckpointManager():
    def __init__(self, model, optimizer=None, ema_model=None, save_dir='', keep_num=10, rank=0, additions={}):
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model
        self.additions = additions
        self.save_dir = save_dir
        self.keep_num = keep_num
        self.rank = rank
        self.ckpts = []
        if self.rank == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.metrics_fp = open(os.path.join(save_dir, 'metrics.csv'), 'a')
            self.metrics_fp.write('epoch,train_loss,test_loss,top1,top5\n')

    def update(self, epoch, metrics, score_key='top1'):
        if self.rank == 0:
            self.metrics_fp.write('{},{},{},{},{}\n'.format(epoch, metrics['train_loss'], metrics['test_loss'], metrics['top1'], metrics['top5']))
            self.metrics_fp.flush()

        score = metrics[score_key]
        insert_idx = 0
        for ckpt_, score_ in self.ckpts:
            if score > score_:
                break
            insert_idx += 1
        if insert_idx < self.keep_num:
            save_path = os.path.join(self.save_dir, 'checkpoint-{}.pth.tar'.format(epoch))
            self.ckpts.insert(insert_idx, [save_path, score])
            if len(self.ckpts) > self.keep_num:
                remove_ckpt = self.ckpts.pop(-1)[0]
                if self.rank == 0:
                    if os.path.exists(remove_ckpt):
                        os.remove(remove_ckpt)
            self._save(save_path, epoch, is_best=(insert_idx == 0))
        else:
            self._save(os.path.join(self.save_dir, 'last.pth.tar'), epoch)
        return self.ckpts

    def _save(self, save_path, epoch, is_best=False):
        if self.rank != 0:
            return
        save_dict = {
            'epoch': epoch,
            'model': self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
            'ema_model': self.ema_model.state_dict() if self.ema_model else None,
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
        }
        for key, value in self.additions.items():
            save_dict[key] = value.state_dict() if hasattr(value, 'state_dict') else value

        torch.save(save_dict, save_path)
        if save_path != os.path.join(self.save_dir, 'last.pth.tar'):
            shutil.copy(save_path, os.path.join(self.save_dir, 'last.pth.tar'))
        if is_best:
            shutil.copy(save_path, os.path.join(self.save_dir, 'best.pth.tar'))

    def load(self, ckpt_path):
        save_dict = torch.load(ckpt_path, map_location='cpu')

        for key, value in self.additions.items():
            if hasattr(value, 'load_state_dict'):
                value.load_state_dict(save_dict[key])
            else:
                self.additions[key] = save_dict[key]

        if 'state_dict' in save_dict and 'model' not in save_dict:
            save_dict['model'] = save_dict['state_dict']
        if isinstance(self.model, DDP):
            missing_keys, unexpected_keys = \
                self.model.module.load_state_dict(save_dict['model'], strict=False)
        else:
            missing_keys, unexpected_keys = \
                self.model.load_state_dict(save_dict['model'], strict=False)
        if len(missing_keys) != 0:
            logger.info(f'Missing keys in source state dict: {missing_keys}')
        if len(unexpected_keys) != 0:
            logger.info(f'Unexpected keys in source state dict: {unexpected_keys}')
        
        if self.ema_model is not None and 'ema_model' in save_dict:
            self.ema_model.load_state_dict(save_dict['ema_model'])
        if self.optimizer is not None and 'optimizer' in save_dict:
            self.optimizer.load_state_dict(save_dict['optimizer'])

        if 'epoch' in save_dict:
            epoch = save_dict['epoch']
        else:
            epoch = -1

        '''avoid memory leak'''
        del save_dict
        torch.cuda.empty_cache()

        return epoch


class AuxiliaryOutputBuffer:

    def __init__(self, model, loss_weight=1.0):
        self.loss_weight = loss_weight
        self.model = model
        self.aux_head = model.module.auxiliary_head
        self._output = None
        self.model.module.module_to_auxiliary.register_forward_hook(lambda net, input, output: self._forward_hook(net, input, output))

    def _forward_hook(self, net, input, output):
        if net.training:
            self._output = self.aux_head(output)

    @property
    def output(self):
        output = self._output
        self._output = None
        return output
