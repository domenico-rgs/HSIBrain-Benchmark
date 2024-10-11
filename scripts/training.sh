#!/bin/bash
conda activate ViT-G

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model ViT --batch-size 8192 --epochs 500 --world-size 8 --num_workers 32 --ngpus 4 --nodes 2 --job-name 'ViT' --comment 'ViT training test'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model ViM --batch-size 8192 --epochs 500 --world-size 8 --num_workers 32 --ngpus 8 --nodes 1 --job-name 'ViM' --comment 'ViM training test'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model HSIMamba --batch-size 8192 --epochs 500 --world-size 8 --num_workers 32 --ngpus 4 --nodes 2 --job-name 'HSIMamba' --comment 'HSIMamba training test'