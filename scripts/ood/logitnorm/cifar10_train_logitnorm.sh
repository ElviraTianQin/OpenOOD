#!/bin/bash
# sh scripts/ood/logitnorm/cifar10_train_logitnorm.sh

# gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
#      awk '{print NR-1, $1}' | sort -k2 -n | head -n1 | cut -d' ' -f1)

# echo "Auto-selected GPU $gpu"

CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_logitnorm.yml \
    configs/preprocessors/base_preprocessor.yml \
    --seed 0
