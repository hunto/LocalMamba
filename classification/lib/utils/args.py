import argparse
import yaml
import torch


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value
            

config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
config_parser.add_argument('-c', '--config', default='', type=str,
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='ImageNet Training')

# Dataset / Model parameters
parser.add_argument('--dataset', default='imagenet', type=str, choices=['cifar10', 'cifar100', 'imagenet'],
                    help='Dataset to use')
parser.add_argument('--data-path', default='', type=str,
                    help='Path to load dataset')
parser.add_argument('--model', default='nas_model', type=str,
                    help='Name of model to train (default: "countception"')
parser.add_argument('--model-config', type=str, default='',
                    help='Path to net config. Used for NAS model.')
parser.add_argument('--resume', default='', type=str,
                    help='Resume the states of model, optimizer, etc. in a checkpoint file')
parser.add_argument('-b', '--batch-size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--val-batch-size-multiplier', type=float, default=1.0,
                    help='batch size of validation data equals to (batch-size * val-batch-size-multiplier)')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--smoothing', default=0.1, type=float,
                    help='Epsilon value of label smoothing')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str,
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float,
                    help='Optimizer Epsilon (default: 1e-8, use opt default)')
parser.add_argument('--opt-no-filter', action='store_true', default=False,
                    help='disable bias and bn filter of weight decay')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--sgd-no-nesterov', action='store_true', default=False,
                    help='set nesterov=False in SGD optimizer')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad-norm', action='store_true', default=False,
                    help='clip gradients of network')
parser.add_argument('--clip-grad-max-norm', type=float, default=5.,
                    help='value of max_norm in clip_grad_norm')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use automatic mixed precision training (torch.cuda.amp)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, 
                    help='LR scheduler (default: "step"')
parser.add_argument('--decay-epochs', type=float, default=3, 
                    help='epoch interval to decay LR')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--warmup-lr', type=float, default=0.0001,
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, 
                    help='minimal learning rate (default: 1e-5)')
parser.add_argument('--epochs', type=int, default=200, 
                    help='number of epochs to train (default: 2)')
parser.add_argument('--warmup-epochs', type=int, default=3, 
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--decay-rate', '--dr', type=float, 
                    help='LR decay rate')
parser.add_argument('--decay_by_epoch', action='store_true', default=False,
                    help='decay LR by epoch, valid only for cosine scheduler')

# Augmentation & regularization parameters
parser.add_argument('--image-mean', type=float, nargs=3, default=None,
                    help='Mean values of image normalization')
parser.add_argument('--image-std', type=float, nargs=3, default=None,
                    help='Std values of image normalization')
parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'],
                    help='Interpolation mode in image resize')
parser.add_argument('--color-jitter', type=float, default=0.,
                    help='Color jitter factor (default: 0.)')
parser.add_argument('--cutout-length', type=int, default=0,
                    help='Cutout length. Only used in CIFAR transforms')
parser.add_argument('--aa', type=str, default=None,
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--reprob', type=float, default=0.,
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--drop', type=float, default=0.0,
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path-rate', type=float, default=0., 
                    help='Drop path rate, (default: 0.)')
parser.add_argument('--drop-path-strategy', type=str, default='const', choices=['const', 'linear'],
                    help='Drop path rate update strategy, default: const')

# Mixup
parser.add_argument('--mixup', type=float, default=0.,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=0.,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50,
                    help='how many batches to wait before logging training status')
parser.add_argument('-j', '--workers', type=int, default=4,
                    help='how many training processes to use (default: 1)')
parser.add_argument('--experiment', default='exp', type=str, 
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--slurm', action='store_true', default=False,
                    help='Use slurm')
if torch.__version__ >= '2.0.0':
    parser.add_argument('--local-rank', default=0, type=int,
                        help='local rank of current process in distributed running')
else:
    parser.add_argument('--local_rank', default=0, type=int,
                        help='local rank of current process in distributed running')
parser.add_argument('--dist-port', default='12345', type=str,
                    help='port for distributed communication')

# KD
parser.add_argument('--kd', type=str, default='',
                    help='Knowledge distillation method. Default: disable')
parser.add_argument('--teacher-model', type=str, default='',
                    help='teacher model name')
parser.add_argument('--teacher-pretrained', action='store_true',
                    help='load pretrained model of teacher')
parser.add_argument('--teacher-no-pretrained', action='store_false', dest='teacher_pretrained')
parser.set_defaults(teacher_pretrained=True)
parser.add_argument('--teacher-ckpt', type=str, default='',
                    help='path to the ckpt of teacher model')
parser.add_argument('--kd-loss-weight', type=float, default=1., 
                    help='weight of kd loss')
parser.add_argument('--ori-loss-weight', type=float, default=1., 
                    help='weight of original loss')
parser.add_argument('--teacher-module', type=str, default='',
                    help='name of the teacher module used in kd. Default (""): use the output of model.')
parser.add_argument('--student-module', type=str, default='',
                    help='name of the student module used in kd. Default (""): use the output of model.')
parser.add_argument('--kd-loss-kwargs', nargs='*', action=ParseKwargs)

# DBB
parser.add_argument('--dbb', action='store_true', default=False,
                    help='Use DBB')

# DyRep
parser.add_argument('--dyrep', action='store_true', default=False,
                    help='Use DyRep')
parser.add_argument('--dyrep-adjust-interval', type=int, default=10,
                    help='how many epochs to rep & dep the dyrep model')
parser.add_argument('--dyrep-max-adjust-epochs', type=int, default=100,
                    help='after how many epochs the dyrep model will be fixed.')
parser.add_argument('--dyrep-recal-bn-iters', type=int, default=20,
                    help='how many iterations for recalibrating the bn states in dyrep')
parser.add_argument('--dyrep-recal-bn-every-epoch', action='store_true', default=False,
                    help='Recal BN after every epoch in DyRep')

# EdgeNN
parser.add_argument('--edgenn-config', type=str, default='',
                    help='path to edgenn config')


def parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    default_dicts = {}
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

            for k, v in cfg.items():
                if isinstance(v, dict):
                    default_dicts[k] = v
                    
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    for k, v in default_dicts.items():
        v.update(args.__dict__[k])
    args.__dict__[k] = v

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


