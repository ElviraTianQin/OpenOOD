#!/bin/bash
# sh scripts/ood/logitnorm/cifar10_test_logitnorm.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
# gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
#      awk '{print NR-1, $1}' | sort -k2 -n | head -n1 | cut -d' ' -f1)

# echo "Auto-selected GPU $gpu"

CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_resnet18_32x32_logitnorm_e100_lr0.1_alpha0.04_default \
   --postprocessor msp \
   --save-score --save-csv
