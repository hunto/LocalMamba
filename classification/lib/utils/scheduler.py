from collections import OrderedDict
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR


def build_scheduler(sched_type, optimizer, warmup_steps, warmup_lr, step_size, decay_rate, total_steps=-1, multiplier=1, steps_per_epoch=1, decay_by_epoch=True, min_lr=1e-5):
    if sched_type == 'step':
        scheduler = StepLR(optimizer, step_size, gamma=decay_rate)
        decay_by_epoch = False
    elif sched_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr)
    elif sched_type == 'linear':
        scheduler = LambdaLR(optimizer, lambda epoch: (total_steps - warmup_steps - epoch) / (total_steps - warmup_steps))
    else:
        raise NotImplementedError(f'Scheduler {sched_type} not implemented.')
    scheduler = GradualWarmupScheduler(optimizer, multiplier=multiplier, total_epoch=warmup_steps, after_scheduler=scheduler, warmup_lr=warmup_lr, step_size=steps_per_epoch, decay_by_epoch=decay_by_epoch)
    return scheduler


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Modified based on: https://github.com/ildoonet/pytorch-gradual-warmup-lr
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
        warmup_lr: warmup learning rate for the first epoch
        step_size: step number in one epoch
        decay_by_epoch: if True, decay lr in after_scheduler after each epoch; otherwise decay after every step
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None, warmup_lr=1e-6, step_size=1, decay_by_epoch=True):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.warmup_lr = warmup_lr
        self.step_size = step_size
        self.finished = False
        if self.total_epoch == 0:
            self.finished = True
            self.total_epoch = -1
        self.decay_by_epoch = decay_by_epoch
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch or self.finished:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [self.warmup_lr + (base_lr - self.warmup_lr) * (float(self.last_epoch // self.step_size * self.step_size) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * (self.last_epoch // self.step_size * self.step_size) / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        if self.last_epoch <= self.total_epoch:
            if self.multiplier == 1.0:
                warmup_lr = [self.warmup_lr + (base_lr - self.warmup_lr) * (float(self.last_epoch // self.step_size * self.step_size) / self.total_epoch) for base_lr in self.base_lrs]
            else:
                warmup_lr = [base_lr * ((self.multiplier - 1.) * (self.last_epoch // self.step_size * self.step_size) / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                if self.decay_by_epoch:
                    self.after_scheduler.step(metrics, (epoch - self.total_epoch - 1) // self.step_size * self.step_size)
                else:
                    self.after_scheduler.step(metrics, epoch - self.total_epoch - 1)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    if self.decay_by_epoch:
                        self.after_scheduler.step((epoch - self.total_epoch - 1) // self.step_size * self.step_size)
                    else:
                        self.after_scheduler.step(epoch - self.total_epoch - 1)
                self._last_lr = self.after_scheduler.get_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


