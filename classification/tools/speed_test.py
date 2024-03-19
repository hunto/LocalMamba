import os
import torch
import torch.nn as nn
import logging
import time
import random
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.builder import build_model
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger
from lib.utils.measure import get_params, get_flops

torch.backends.cudnn.benchmark = True

'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    args, args_text = parse_args()
    args.input_shape = (3, 224, 224)

    '''fix random seed'''
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    '''build model'''
    model = build_model(args, args.model)
    logger.info(
        f'Model {args.model} created, params: {get_params(model)}, '
        f'FLOPs: {get_flops(model, input_shape=args.input_shape)}')

    # Diverse Branch Blocks
    if args.dbb:
        # convert 3x3 convs to dbb blocks
        from lib.models.utils.dbb_converter import convert_to_dbb
        convert_to_dbb(model)
        logger.info(model)
        logger.info(
            f'Converted to DBB blocks, model params: {get_params(model)}, '
            f'FLOPs: {get_flops(model, input_shape=args.input_shape)}')

    speed_test(model, batch_size=args.batch_size, input_shape=args.input_shape)


def speed_test(model, warmup_iters=100, n_iters=1000, batch_size=128, input_shape=(3, 224, 224), device='cuda'):
    device = torch.device(device)
    model.to(device)
    model.eval()
    x = torch.randn((batch_size, *input_shape), device=device)

    with torch.no_grad():
        for _ in range(warmup_iters):
            model(x)
        logger.info('Start measuring speed.')
        torch.cuda.synchronize()
        t = time.time()
        for i in range(n_iters):
            model(x)
        torch.cuda.synchronize()
        total_time = time.time() - t
        total_samples = batch_size * n_iters
        speed = total_samples / total_time
        logger.info(f'Done, n_iters: {n_iters}, batch size: {batch_size}, image shape: {input_shape}')
        logger.info(f'total time: {total_time} s, total samples: {total_samples}, throughput: {speed:.3f} samples/second.')


if __name__ == '__main__':
    main()
