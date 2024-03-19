#!/usr/bin/env bash
ENTRY=$1
PARTITION=$2
GPUS=$3
CONFIG=$4
MODEL=$5
PY_ARGS=${@:6}

N=${GPUS}
if [ ${GPUS} -gt 8 ]
then
    echo "multi machine"
    N=8
fi

set -x

PYTHONPATH=$PWD:$PYTHONPATH PYTHONWARNINGS=ignore GLOG_logtostderr=-1 GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
    srun --mpi=pmi2 --job-name train --partition=${PARTITION} -n${GPUS} --gres=gpu:${N} --ntasks-per-node=${N} \
        python -u ${ENTRY} -c ${CONFIG} --model ${MODEL} --slurm ${PY_ARGS}
