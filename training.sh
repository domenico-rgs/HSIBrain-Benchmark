#!/bin/bash

EPOCHS=300

#Madrid
DATASET='madrid'
DPATH='/home/ragusa/ViT-G/datasets/Madrid/hsi/'
GPATH='/home/ragusa/ViT-G/datasets/Madrid/gt/'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size 8192 --epochs $EPOCHS --world-size 8 --num_workers 1 --ngpus 4 --nodes 2 --job-name 'ViT-M' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --easyAtt True --batch-size 8192 --epochs $EPOCHS --world-size 8 --num_workers 1 --ngpus 8 --nodes 1 --job-name 'ViT+EA-M' --comment 'ViT con EasyAttention training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size 8192 --epochs $EPOCHS --caf True --world-size 8 --num_workers 1 --ngpus 4 --nodes 2 --job-name 'SF-M' --comment 'SpectralFormer training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --caf True --easyAtt True --batch-size 8192 --epochs $EPOCHS --world-size 8 --num_workers 1 --ngpus 8 --nodes 1 --job-name 'SF+EA-M' --comment 'SpectralFormer con EasyAttention training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size 8192 --epochs $EPOCHS --world-size 8 --num_workers 1 --ngpus 4 --nodes 2 --job-name 'ViM-M' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model HSIMamba --batch-size 8192 --epochs $EPOCHS --world-size 8 --num_workers 1 --ngpus 8 --nodes 1 --job-name 'HSIMamba-M' --comment 'HSIMamba training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model HiT --batch-size 8192 --epochs $EPOCHS --world-size 8 --num_workers 1 --ngpus 4 --nodes 2 --job-name 'HiT-M' --comment 'HiT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH

#Las Palmas
DATASET='LP'
DPATH='/home/ragusa/ViT-G/datasets/LP/hsi/'
GPATH='/home/ragusa/ViT-G/datasets/LP/gt/'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size 8192 --epochs $EPOCHS --world-size 8 --num_workers 1 --ngpus 4 --nodes 2 --job-name 'ViT-LP' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --easyAtt True --batch-size 8192 --epochs $EPOCHS --world-size 8 --num_workers 1 --ngpus 8 --nodes 1 --job-name 'ViT+EA-LP' --comment 'ViT con EasyAttention training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size 8192 --epochs $EPOCHS --caf True --world-size 8 --num_workers 1 --ngpus 4 --nodes 2 --job-name 'SF-LP' --comment 'SpectralFormer training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --caf True --easyAtt True --batch-size 8192 --epochs $EPOCHS --world-size 8 --num_workers 1 --ngpus 8 --nodes 1 --job-name 'SF+EA-LP' --comment 'SpectralFormer con EasyAttention training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size 8192 --epochs $EPOCHS --world-size 8 --num_workers 1 --ngpus 4 --nodes 2 --job-name 'ViM-LP' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model HSIMamba --batch-size 8192 --epochs $EPOCHS --world-size 8 --num_workers 1 --ngpus 8 --nodes 1 --job-name 'HSIMamba-LP' --comment 'HSIMamba training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model HiT --batch-size 8192 --epochs $EPOCHS --world-size 8 --num_workers 1 --ngpus 4 --nodes 2 --job-name 'HiT-LP' --comment 'HiT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH
