#!/usr/bin/env bash
PARTITION=$1
GPUS=$2
CONFIG=$3
MODEL=$4
PY_ARGS=${@:5}

N=${GPUS}
if [ ${GPUS} -gt 8 ]
then
    echo "multi machine"
    N=8
fi

set -x

PYTHONPATH=$PWD:$PYTHONPATH PYTHONWARNINGS=ignore GLOG_logtostderr=-1 GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
    srun --mpi=pmi2 --job-name train --partition=${PARTITION} -n${GPUS} --gres=gpu:${N} --ntasks-per-node=${N} \
        python -u tools/train.py -c ${CONFIG} --model ${MODEL} --slurm ${PY_ARGS}
