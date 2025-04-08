#!/bin/bash
# sh scripts/ood/rotpred/cifar10_test_rotpred.sh

python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_bootood_net_bootood_e100_lr0.1_default \
   --postprocessor bootood \
   --save-score --save-csv