import os
import time
import shutil
import logging
import subprocess
import torch


def init_dist(args):
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.slurm:
        args.distributed = True
    if not args.distributed:
        # task with single GPU also needs to use distributed module
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = '0'
        args.local_rank = 0
        args.distributed = True

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        if args.slurm:
            # processes are created with slurm
            proc_id = int(os.environ['SLURM_PROCID'])
            ntasks = int(os.environ['SLURM_NTASKS'])
            node_list = os.environ['SLURM_NODELIST']
            num_gpus = torch.cuda.device_count()
            addr = subprocess.getoutput(
                f'scontrol show hostname {node_list} | head -n1')
            os.environ['MASTER_ADDR'] = addr
            os.environ['WORLD_SIZE'] = str(ntasks)
            args.local_rank = proc_id % num_gpus
            os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
            os.environ['RANK'] = str(proc_id)
            print(f'Using slurm with master node: {addr}, rank: {proc_id}, world size: {ntasks}')
        else:
            addr = os.environ['MASTER_ADDR']
            ntasks = os.environ['WORLD_SIZE']
            proc_id = os.environ['RANK']
            args.local_rank = int(os.environ['LOCAL_RANK'])
            print(f'Using torch.distributed with master node: {addr}, rank: {proc_id}, local_rank: {args.local_rank} world size: {ntasks}')


        #os.environ['MASTER_PORT'] = args.dist_port
        args.device = 'cuda:%d' % args.local_rank
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        # if not args.slurm:
        torch.cuda.set_device(args.local_rank)
        print(f'Training in distributed model with multiple processes, 1 GPU per process. Process {args.rank}, total {args.world_size}.')
    else:
        print('Training with a single process on 1 GPU.')


# create logger file handler for rank 0,
# ignore the outputs of the other ranks
def init_logger(args):
    logger = logging.getLogger()
    if args.rank == 0:
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(args.exp_dir, f'log_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.txt'))
        fh.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%y-%m-%d %H:%M:%S'))
        logger.addHandler(fh)
        logger.info(f'Experiment directory: {args.exp_dir}')

    else:
        logger.setLevel(logging.ERROR)
