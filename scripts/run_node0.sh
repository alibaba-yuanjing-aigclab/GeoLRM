#!/bin/bash

export NCCL_DEBUG=INFO
export MASTER_PORT=11901
export MASTER_ADDR=192.168.3.15
export WORLD_SIZE=4
export NODE_RANK=0
python train.py --base configs/geolrm-train.yaml --num_nodes 4 --gpus 0,1,2,3,4,5,6,7
