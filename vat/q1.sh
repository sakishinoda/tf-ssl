#!/usr/bin/env bash
python train_semisup_mnist.py --id "MlpVat_labeled-1000_seed-1" --seed 1 --which_gpu 1 --batch_size 100 --ul_batch_size 250 --num_epochs 100 --learning_rate 0.002 --num_labeled 1000
python train_semisup_mnist.py --id "MlpVat_labeled-1000_seed-100" --seed 100 --which_gpu 1 --batch_size 100 --ul_batch_size 250 --num_epochs 100 --learning_rate 0.002 --num_labeled 1000
python train_semisup_mnist.py --id "MlpVat_labeled-1000_seed-11" --seed 11 --which_gpu 1 --batch_size 100 --ul_batch_size 250 --num_epochs 100 --learning_rate 0.002 --num_labeled 1000
python train_semisup_mnist.py --id "MlpVat_labeled-1000_seed-111" --seed 111 --which_gpu 1 --batch_size 100 --ul_batch_size 250 --num_epochs 100 --learning_rate 0.002 --num_labeled 1000
python train_semisup_mnist.py --id "MlpVat_labeled-1000_seed-1111" --seed 1111 --which_gpu 1 --batch_size 100 --ul_batch_size 250 --num_epochs 100 --learning_rate 0.002 --num_labeled 1000