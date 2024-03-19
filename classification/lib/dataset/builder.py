import os
import re
import torch
import torchvision.datasets as datasets

from .dataset import ImageNetDataset
from .dataloader import fast_collate, DataPrefetcher
from .mixup import Mixup
from . import transform


def _check_torch_version(target='1.7.0'):
    if torch.__version__ == 'parrots':
        return False
    version = re.match('([\d.])*', torch.__version__).group()
    target = re.match('([\d.])*', target).group()
    major, minor, patch = [int(x) for x in version.split('.')[:3]]
    t_major, t_minor, t_patch = [int(x) for x in target.split('.')[:3]]
    if major > t_major:
        return True
    elif major == t_major:
        if minor > t_minor:
            return True
        elif minor == t_minor:
            if patch >= t_patch:
                return True
    return False


# for pytorch>=1.7.0, we add persistent_workers=True in 
# dataloader params
if _check_torch_version('1.7.0'):
    _LOADER_PARAMS = dict(persistent_workers=True)
else:
    _LOADER_PARAMS = dict()


def build_dataloader(args):
    # pre-configuration for the dataset
    if args.dataset == 'imagenet':
        args.data_path = 'data/imagenet' if args.data_path == '' else args.data_path
        args.num_classes = 1000
        args.input_shape = (3, 224, 224)
    elif args.dataset == 'cifar10':
        args.data_path = 'data/cifar' if args.data_path == '' else args.data_path
        args.num_classes = 10
        args.input_shape = (3, 32, 32)
    elif args.dataset == 'cifar100':
        args.data_path = 'data/cifar' if args.data_path == '' else args.data_path
        args.num_classes = 100
        args.input_shape = (3, 32, 32)

    # train
    if args.dataset == 'imagenet':
        train_transforms_l, train_transforms_r = transform.build_train_transforms(
            args.aa, args.color_jitter, args.reprob, args.remode, args.interpolation, args.image_mean, args.image_std)
        train_dataset = ImageNetDataset(
            os.path.join(args.data_path, 'train'), os.path.join(args.data_path, 'meta/train.txt'), transform=train_transforms_l)
    elif args.dataset == 'cifar10':
        train_transforms_l, train_transforms_r = transform.build_train_transforms_cifar10(
            args.cutout_length, args.image_mean, args.image_std)
        train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=train_transforms_l)
    elif args.dataset == 'cifar100':
        train_transforms_l, train_transforms_r = transform.build_train_transforms_cifar10(
            args.cutout_length, args.image_mean, args.image_std)
        train_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, download=True, transform=train_transforms_l)

    # mixup
    mixup_active = args.mixup > 0. or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_transform = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob,
                                switch_prob=args.mixup_switch_prob, mode=args.mixup_mode, label_smoothing=args.smoothing, num_classes=args.num_classes)
    else:
        mixup_transform = None

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, 
        pin_memory=False, sampler=train_sampler, collate_fn=fast_collate, drop_last=True, **_LOADER_PARAMS)
    train_loader = DataPrefetcher(train_loader, train_transforms_r, mixup_transform)

    # val
    if args.dataset == 'imagenet':
        val_transforms_l, val_transforms_r = transform.build_val_transforms(args.interpolation, args.image_mean, args.image_std)
        val_dataset = ImageNetDataset(os.path.join(args.data_path, 'val'), os.path.join(args.data_path, 'meta/val.txt'), transform=val_transforms_l)
    elif args.dataset == 'cifar10':
        val_transforms_l, val_transforms_r = transform.build_val_transforms_cifar10(args.image_mean, args.image_std)
        val_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=val_transforms_l)
    elif args.dataset == 'cifar100':
        val_transforms_l, val_transforms_r = transform.build_val_transforms_cifar10(args.image_mean, args.image_std)
        val_dataset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=val_transforms_l)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(args.batch_size * args.val_batch_size_multiplier), 
        shuffle=False, num_workers=args.workers, pin_memory=False, 
        sampler=val_sampler, collate_fn=fast_collate, **_LOADER_PARAMS)
    val_loader = DataPrefetcher(val_loader, val_transforms_r)

    return train_dataset, val_dataset, train_loader, val_loader
