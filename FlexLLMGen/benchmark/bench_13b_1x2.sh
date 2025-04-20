#!/bin/bash

MY_IPADDR=$(hostname -i)
all_hosts=$MY_IPADDR
N_GPUS=2
N_CORES_PER_GPU=6

PYTHON_EXEC=$CONDA_PREFIX/bin/python
PYTHON_SCRIPT=flexgen_tp.dist_flex_opt

pgrep -fl python | awk '!/dist_flex_opt\.py/{print $1}' | xargs sudo kill

set -x

mpirun \
  --mca btl sm,self \
  --map-by ppr:2:node:pe=$N_CORES_PER_GPU \
  --bind-to core -x OMP_NUM_THREADS=$N_CORES_PER_GPU \
  /home/cc/LLM/bin/python -m flexgen_tp.dist_flex_opt \
    --head-ip $MY_IPADDR \
    --port 7777 \
    --use-mpi \
    --model facebook/opt-13b \
    --gpu-batch-size 8 \
    --percent 100 0 100 0 100 0 \
    --comm-device cpu \
    --prompt-len 2 --gen-len 128 \
    --path _DUMMY_ 
