#!/usr/bin/env bash

python train_semisup_mnist.py --id "DeepMlpVat_beta1-0.5_epsilon-5.0_epochs-400_decay-200_seed-1" --seed 1 --epsilon 5.0 --which_gpu 2 --num_epochs 400 --epoch_decay_start 200
python train_semisup_mnist.py --id "DeepMlpVat_beta1-0.5_epsilon-5.0_epochs-400_decay-200_seed-100" --seed 100 --epsilon 5.0 --which_gpu 2 --num_epochs 400 --epoch_decay_start 200
python train_semisup_mnist.py --id "DeepMlpVat_beta1-0.5_epsilon-5.0_epochs-400_decay-200_seed-11" --seed 11 --epsilon 5.0 --which_gpu 2 --num_epochs 400 --epoch_decay_start 200
python train_semisup_mnist.py --id "DeepMlpVat_beta1-0.5_epsilon-5.0_epochs-400_decay-200_seed-111" --seed 111 --epsilon 5.0 --which_gpu 2 --num_epochs 400 --epoch_decay_start 200
python train_semisup_mnist.py --id "DeepMlpVat_beta1-0.5_epsilon-5.0_epochs-400_decay-200_seed-1111" --seed 1111 --epsilon 5.0 --which_gpu 2 --num_epochs 400 --epoch_decay_start 200
