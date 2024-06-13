#! /bin/sh
CUDA_VISIBLE_DEVICES=0 python train.py --camera realsense --log_dir logs/log_gengrasp --batch_size 2 --dataset_root /home/LAB/r-yanghongyu/data/graspnet