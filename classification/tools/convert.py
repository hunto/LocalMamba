import os
import torch
import torch.nn as nn
import logging
import time
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.builder import build_model
from lib.models.loss import CrossEntropyLabelSmooth
from lib.models.utils.dbb.dbb_block import DiverseBranchBlock
from lib.dataset.builder import build_dataloader
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger
from lib.utils.misc import accuracy, AverageMeter, CheckpointManager
from lib.utils.model_ema import ModelEMA
from lib.utils.measure import get_params, get_flops

torch.backends.cudnn.benchmark = True
'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    args, args_text = parse_args()
    assert args.resume != ''
    args.exp_dir = f'{os.path.dirname(args.resume)}/convert'

    '''distributed'''
    init_dist(args)
    init_logger(args)

    '''build dataloader'''
    train_dataset, val_dataset, train_loader, val_loader = \
        build_dataloader(args)

    '''build model'''
    if args.smoothing == 0.:
        loss_fn = nn.CrossEntropyLoss().cuda()
    else:
        loss_fn = CrossEntropyLabelSmooth(num_classes=args.num_classes,
                                          epsilon=args.smoothing).cuda()

    model = build_model(args)
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

    model.cuda()
    model = DDP(model,
                device_ids=[args.local_rank],
                find_unused_parameters=False)

    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None

    '''dyrep'''
    if args.dyrep:
        from lib.models.utils.dyrep import DyRep
        dyrep = DyRep(
            model.module,
            None)
        logger.info('Init DyRep done.')
    else:
        dyrep = None

    '''resume'''
    ckpt_manager = CheckpointManager(model,
                                     ema_model=model_ema,
                                     save_dir=args.exp_dir,
                                     rank=args.rank,
                                     additions={
                                         'dyrep': dyrep
                                     })

    if args.resume:
        epoch = ckpt_manager.load(args.resume)
        if args.dyrep:
            model = DDP(model.module,
                        device_ids=[args.local_rank],
                        find_unused_parameters=True)
        logger.info(
            f'Resume ckpt {args.resume} done, '
            f'epoch {epoch}'
        )
    else:
        epoch = 0

    # validate
    test_metrics = validate(args, epoch, model, val_loader, loss_fn)
    # convert dyrep / dbb model to inference model
    for m in model.module.modules():
        if isinstance(m, DiverseBranchBlock):
            m.switch_to_deploy()
    logger.info(str(model))
    logger.info(
        f'Converted DBB / DyRep model to inference model, params: {get_params(model)}, '
        f'FLOPs: {get_flops(model, input_shape=args.input_shape)}')
    test_metrics = validate(args, epoch, model, val_loader, loss_fn)

    '''save converted checkpoint'''
    if args.rank == 0:
        save_path = os.path.join(args.exp_dir, 'model.ckpt')
        torch.save(model.module.state_dict(), save_path)
        logger.info(f'Saved converted model checkpoint into {save_path} .')


def validate(args, epoch, model, loader, loss_fn, log_suffix=''):
    loss_m = AverageMeter(dist=True)
    top1_m = AverageMeter(dist=True)
    top5_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    model.eval()
    for batch_idx, (input, target) in enumerate(loader):
        with torch.no_grad():
            output = model(input)
            loss = loss_fn(output, target)

        top1, top5 = accuracy(output, target, topk=(1, 5))
        loss_m.update(loss.item(), n=input.size(0))
        top1_m.update(top1 * 100, n=input.size(0))
        top5_m.update(top5 * 100, n=input.size(0))

        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Test{}: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'Top-1: {top1.val:.3f}% ({top1.avg:.3f}%) '
                        'Top-5: {top5.val:.3f}% ({top5.avg:.3f}%) '
                        'Time: {batch_time.val:.2f}s'.format(
                            log_suffix,
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m))
        start_time = time.time()

    return {'test_loss': loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}


if __name__ == '__main__':
    main()
