#!/bin/bash
# sh scripts/c_covid/0_covid_train.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
-w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/covid/covid.yml \
configs/networks/resnet18_224x224.yml \
configs/pipelines/train/baseline.yml \
--network.pretrained True \
--network.checkpoint ./results/covid_resnet18_224x224_base_e200_lr0.001/resnet18-5c106cde.pth \
--optimizer.num_epochs 50 \
--optimizer.lr 0.001 \
--optimizer.weight_decay 0.005 \
--num_workers 8
