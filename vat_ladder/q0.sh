#!/usr/bin/env bash

# GPU 0

python vat_ladder.py --batch_size 100 --beta1 0.9 --beta1_during_decay 0.5 --decay_start 0.5 --encoder_layers 784-1000-500-250-250-250-10 --encoder_noise_sd 0.3 --end_epoch 200 --id n-test --initial_learning_rate 0.002 --logdir logs/ --lr_decay_frequency 10 --lw 5.0-0.5-0.05-0.05-0.05-0.05-0.05 --num_labeled 100 --num_power_iterations 1 --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --seed 8340 --static_bn --test_frequency_in_epochs 10 --ul_batch_size 250 --validation --which_gpu 0 --do_not_save --model nlw

python vat_ladder.py --batch_size 100 --beta1 0.9 --beta1_during_decay 0.5 --decay_start 0.5 --encoder_layers 784-1000-500-250-250-250-10 --encoder_noise_sd 0.3 --end_epoch 200 --id nlw-test --initial_learning_rate 0.002 --logdir logs/ --lr_decay_frequency 10 --epsilon 5.0 --num_labeled 100 --num_power_iterations 1 --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --seed 8340 --static_bn --test_frequency_in_epochs 10 --ul_batch_size 250 --validation --which_gpu 0 --do_not_save --model n