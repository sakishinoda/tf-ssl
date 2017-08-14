#!/usr/bin/env bash

python train_semisup_mnist.py --id "MlpVat_default_seed-100" --seed 100 --which_gpu 1
python train_semisup_mnist.py --id "MlpVat_default_seed-11" --seed 11 --which_gpu 1
python train_semisup_mnist.py --id "MlpVat_default_seed-111" --seed 111 --which_gpu 1
python train_semisup_mnist.py --id "MlpVat_default_seed-1111" --seed 1111 --which_gpu 1