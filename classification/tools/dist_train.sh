#!/bin/bash
GPUS=$1
CONFIG=$2
MODEL=$3
PY_ARGS=${@:4}

MASTER_PORT=29500

set -x

# NOTE: This script only supports run on single machine and single (multiple) GPUs.
#       You may need to modify it to support multi-machine multi-card training on your distributed platform.

python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    tools/train.py -c ${CONFIG} --model ${MODEL} ${PY_ARGS}
