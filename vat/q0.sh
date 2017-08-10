#!/usr/bin/env bash
python train_semisup_mnist.py --id "MlpVat_seed-1" --seed 1 --which_gpu 0 --batch_size 100 --ul_batch_size 250 --num_epochs 100 --learning_rate 0.002
python train_semisup_mnist.py --id "MlpVat_seed-100" --seed 100 --which_gpu 0 --batch_size 100 --ul_batch_size 250 --num_epochs 100 --learning_rate 0.002
python train_semisup_mnist.py --id "MlpVat_seed-11" --seed 11 --which_gpu 0 --batch_size 100 --ul_batch_size 250 --num_epochs 100 --learning_rate 0.002
python train_semisup_mnist.py --id "MlpVat_seed-111" --seed 111 --which_gpu 0 --batch_size 100 --ul_batch_size 250 --num_epochs 100 --learning_rate 0.002
python train_semisup_mnist.py --id "MlpVat_seed-1111" --seed 1111 --which_gpu 0 --batch_size 100 --ul_batch_size 250 --num_epochs 100 --learning_rate 0.002
