#!/usr/bin/env bash

python train_semisup_mnist.py --id "DeepMlpVat_beta1-0.5_epsilon-5.0_epochs-400_decay-200_seed-8340" --seed 8340 --epsilon 5.0 --which_gpu 2 --num_epochs 400 --epoch_decay_start 200
python train_semisup_mnist.py --id "DeepMlpVat_beta1-0.5_epsilon-5.0_epochs-400_decay-200_seed-8794" --seed 8794 --epsilon 5.0 --which_gpu 2 --num_epochs 400 --epoch_decay_start 200
python train_semisup_mnist.py --id "DeepMlpVat_beta1-0.5_epsilon-5.0_epochs-400_decay-200_seed-2773" --seed 2773 --epsilon 5.0 --which_gpu 2 --num_epochs 400 --epoch_decay_start 200
python train_semisup_mnist.py --id "DeepMlpVat_beta1-0.5_epsilon-5.0_epochs-400_decay-200_seed-967" --seed 967 --epsilon 5.0 --which_gpu 2 --num_epochs 400 --epoch_decay_start 200
python train_semisup_mnist.py --id "DeepMlpVat_beta1-0.5_epsilon-5.0_epochs-400_decay-200_seed-2368" --seed 2368 --epsilon 5.0 --which_gpu 2 --num_epochs 400 --epoch_decay_start 200
