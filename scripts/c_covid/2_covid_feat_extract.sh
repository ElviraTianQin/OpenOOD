#!/bin/bash
# sh scripts/c_covid/2_covid_feat_extract.sh

GPU=1
CPU=1
node=68
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/covid/covid.yml \
configs/networks/resnet18_224x224.yml \
configs/pipelines/test/feat_extract.yml \
--network.checkpoint ./results/covid_resnet18_224x224_base_e100_lr0.001/best_epoch80_acc0.7435897435897436.ckpt \
--num_workers 4
