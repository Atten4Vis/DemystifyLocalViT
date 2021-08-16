#!/usr/bin/env bash

NCCL_SOCKET_IFNAME=ib0

MASTER_IP=${MASTER_IP}
MASTER_PORT=12345
NODE_RANK=${OMPI_COMM_WORLD_RANK} && echo NODE_RANK: ${NODE_RANK}
PER_NODE_GPU=8 && echo PER_NODE_GPU: ${PER_NODE_GPU}
NUM_NODE=${OMPI_COMM_WORLD_SIZE} && echo NUM_NODE: ${NUM_NODE}
    
MKL_THREADING_LAYER=GNU python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_IP \
    --master_port=$MASTER_PORT \
    main.py \
    --cfg ./configs/dynamic_dwnet_base_patch4_window7_224.yaml \
    --data-path "/path/to/imagenet" \
    --output "output/dynamic_dwnet_base_patch4_window7_224" \
    --data-set IMNET \
    --batch-size 64 \
    --amp-opt-level O0