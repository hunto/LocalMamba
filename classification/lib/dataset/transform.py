import torch
import torchvision.transforms as transforms
from PIL import Image
from . import augment_ops


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

CIFAR_DEFAULT_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR_DEFAULT_STD = (0.24703233, 0.24348505, 0.26158768)


def build_train_transforms(aa_config_str="rand-m9-mstd0.5", color_jitter=None, 
                           reprob=0., remode='pixel', interpolation='bilinear', mean=None, std=None):
    mean = IMAGENET_DEFAULT_MEAN if mean is None else mean
    std = IMAGENET_DEFAULT_STD if std is None else std
    trans_l = []
    trans_r = []
    trans_l.extend([
                augment_ops.RandomResizedCropAndInterpolation(224, interpolation=interpolation),
                transforms.RandomHorizontalFlip()])
    if aa_config_str is not None and aa_config_str != '':
        if interpolation == 'bilinear':
           aa_interpolation = Image.BILINEAR
        elif interpolation == 'bicubic':
           aa_interpolation = Image.BICUBIC
        else:
           raise RuntimeError(f'Interpolation mode {interpolation} not found.')

        aa_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([round(x * 255) for x in IMAGENET_DEFAULT_MEAN]),
            interpolation=aa_interpolation
        )
        trans_l.append(augment_ops.rand_augment_transform(aa_config_str, aa_params))
    elif color_jitter != 0 and color_jitter is not None:
        # enable color_jitter when not using AA
        trans_l.append(transforms.ColorJitter(color_jitter, color_jitter, color_jitter))
    trans_l.append(augment_ops.ToNumpy())
    
    trans_r.append(augment_ops.Normalize(mean=[x * 255 for x in mean],
                                        std=[x * 255 for x in std]))
    if reprob > 0:
        trans_r.append(augment_ops.RandomErasing(reprob, mode=remode, max_count=1, num_splits=0, device='cuda'))
    return transforms.Compose(trans_l), transforms.Compose(trans_r)
    

def build_val_transforms(interpolation='bilinear', mean=None, std=None):
    mean = IMAGENET_DEFAULT_MEAN if mean is None else mean
    std = IMAGENET_DEFAULT_STD if std is None else std
    if interpolation == 'bilinear':
        interpolation = Image.BILINEAR
    elif interpolation == 'bicubic':
        interpolation = Image.BICUBIC
    else:
       raise RuntimeError(f'Interpolation mode {interpolation} not found.')

    trans_l = transforms.Compose([
                  transforms.Resize(256, interpolation=interpolation),
                  transforms.CenterCrop(224),
                  augment_ops.ToNumpy()
              ])
    trans_r = transforms.Compose([
                  augment_ops.Normalize(mean=[x * 255 for x in mean],
                                        std=[x * 255 for x in std])
              ])
    return trans_l, trans_r


def build_train_transforms_cifar10(cutout_length=0., mean=None, std=None):
    mean = CIFAR_DEFAULT_MEAN if mean is None else mean
    std = CIFAR_DEFAULT_STD if std is None else std
    trans_l = transforms.Compose([
                  transforms.RandomCrop(32, padding=4),
                  transforms.RandomHorizontalFlip(),
                  augment_ops.ToNumpy()
              ])
    trans_r = [
        augment_ops.Normalize(mean=[x * 255 for x in mean],
                              std=[x * 255 for x in std]) 
    ]
    if cutout_length != 0:
        trans_r.append(augment_ops.Cutout(length=cutout_length))
    trans_r = transforms.Compose(trans_r)
    return trans_l, trans_r


def build_val_transforms_cifar10(mean=None, std=None):
    mean = CIFAR_DEFAULT_MEAN if mean is None else mean
    std = CIFAR_DEFAULT_STD if std is None else std
    trans_l = transforms.Compose([
                  augment_ops.ToNumpy()
              ])
    trans_r = transforms.Compose([
                  augment_ops.Normalize(mean=[x * 255 for x in mean],
                                        std=[x * 255 for x in std]) 
              ])
    return trans_l, trans_r



