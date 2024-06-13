#! /bin/sh
CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_phy  --checkpoint_path logs/log_phy/checkpoint.tar --collision_thresh 0 --camera realsense --dataset_root /home/LAB/r-yanghongyu/data/graspnet

