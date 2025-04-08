#!/bin/bash
# sh scripts/ood/bootood/cifar10_train_bootood.sh

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml \
    --trainer.name bootood \
    --output_dir ./results/cifar10_bootood_net_default \
    --seed 0
