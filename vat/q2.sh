#!/usr/bin/env bash

python train_semisup_mnist.py --id "MlpVat_epsilon-0.1_seed-1" --seed 1 --which_gpu 2 --epsilon 0.1
python train_semisup_mnist.py --id "MlpVat_epsilon-0.1_seed-100" --seed 100 --which_gpu 2 --epsilon 0.1
python train_semisup_mnist.py --id "MlpVat_epsilon-0.1_seed-11" --seed 11 --which_gpu 2 --epsilon 0.1
python train_semisup_mnist.py --id "MlpVat_epsilon-0.1_seed-111" --seed 111 --which_gpu 2 --epsilon 0.1
python train_semisup_mnist.py --id "MlpVat_epsilon-0.1_seed-1111" --seed 1111 --which_gpu 2 --epsilon 0.1
