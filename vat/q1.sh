#!/usr/bin/env bash

python train_semisup_mnist.py --id "labeled-1000_seed-1" --seed 1 --epsilon 5.0 --which_gpu 1 --num_epochs 400 --epoch_decay_start 200 --num_labeled 1000
python train_semisup_mnist.py --id "labeled-1000_seed-100" --seed 100 --epsilon 5.0 --which_gpu 1 --num_epochs 400 --epoch_decay_start 200 --num_labeled 1000
python train_semisup_mnist.py --id "labeled-1000_seed-11" --seed 11 --epsilon 5.0 --which_gpu 1 --num_epochs 400 --epoch_decay_start 200 --num_labeled 1000
python train_semisup_mnist.py --id "labeled-1000_seed-111" --seed 111 --epsilon 5.0 --which_gpu 1 --num_epochs 400 --epoch_decay_start 200 --num_labeled 1000
python train_semisup_mnist.py --id "labeled-1000_seed-1111" --seed 1111 --epsilon 5.0 --which_gpu 1 --num_epochs 400 --epoch_decay_start 200 --num_labeled 1000
