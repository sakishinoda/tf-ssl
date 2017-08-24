#!/usr/bin/env bash

nohup python -u hyperopt.py  --batch_size 50 --beta1 0.9 --beta1_during_decay 0.9 --corrupt_sd 0.3 --dataset mnist --decay_start 1.0 --encoder_layers 784-1000-500-250-250-250-10 --end_epoch 25 --epsilon 5.0 --id nlw_labeled-50 --initial_learning_rate 0.002 --logdir logs/ --lr_decay_frequency 10 --lrelu_a 0.1 --model nlw --num_labeled 50 --num_power_iters 3 --rc_weights 0-0-0-0-0-0-10 --seed 1 --static_bn 0.99 --test_frequency_in_epochs 10 --ul_batch_size 150 --vadv_sd 0.5 --validation 1000 --which_gpu 0 --xi 1e-6 > hyperopt_nlw_labeled-50.log &

