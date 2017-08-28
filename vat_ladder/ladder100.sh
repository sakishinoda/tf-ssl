#!/usr/bin/env bash


python vat_ladder.py  --batch_size 100 --beta1 0.5 --beta1_during_decay 0.5 --corrupt_sd 0.2 --dataset mnist --decay_start 0.8 --encoder_layers 1000-500-250-250-250-10 --end_epoch 250 --epsilon 1.0 --id full_ladder_labeled-100 --initial_learning_rate 0.002 --logdir logs/ --lr_decay_frequency 10 --lrelu 0.0 --lrelu_a 0.1 --model ladder --num_labeled 100 --num_power_iters 3 --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --seed 7698 --static_bn 0.99 --test_frequency_in_epochs 5 --ul_batch_size 250 --vadv_sd 0.5 --validation 0 --which_gpu 1 --xi 1e-6 

python vat_ladder.py  --batch_size 100 --beta1 0.5 --beta1_during_decay 0.5 --corrupt_sd 0.2 --dataset mnist --decay_start 0.8 --encoder_layers 1000-500-250-250-250-10 --end_epoch 250 --epsilon 1.0 --id full_ladder_labeled-100 --initial_learning_rate 0.002 --logdir logs/ --lr_decay_frequency 10 --lrelu 0.0 --lrelu_a 0.1 --model ladder --num_labeled 100 --num_power_iters 3 --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --seed 2049 --static_bn 0.99 --test_frequency_in_epochs 5 --ul_batch_size 250 --vadv_sd 0.5 --validation 0 --which_gpu 1 --xi 1e-6 

python vat_ladder.py  --batch_size 100 --beta1 0.5 --beta1_during_decay 0.5 --corrupt_sd 0.2 --dataset mnist --decay_start 0.8 --encoder_layers 1000-500-250-250-250-10 --end_epoch 250 --epsilon 1.0 --id full_ladder_labeled-100 --initial_learning_rate 0.002 --logdir logs/ --lr_decay_frequency 10 --lrelu 0.0 --lrelu_a 0.1 --model ladder --num_labeled 100 --num_power_iters 3 --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --seed 1496 --static_bn 0.99 --test_frequency_in_epochs 5 --ul_batch_size 250 --vadv_sd 0.5 --validation 0 --which_gpu 1 --xi 1e-6 

python vat_ladder.py  --batch_size 100 --beta1 0.5 --beta1_during_decay 0.5 --corrupt_sd 0.2 --dataset mnist --decay_start 0.8 --encoder_layers 1000-500-250-250-250-10 --end_epoch 250 --epsilon 1.0 --id full_ladder_labeled-100 --initial_learning_rate 0.002 --logdir logs/ --lr_decay_frequency 10 --lrelu 0.0 --lrelu_a 0.1 --model ladder --num_labeled 100 --num_power_iters 3 --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --seed 748 --static_bn 0.99 --test_frequency_in_epochs 5 --ul_batch_size 250 --vadv_sd 0.5 --validation 0 --which_gpu 1 --xi 1e-6 

python vat_ladder.py  --batch_size 100 --beta1 0.5 --beta1_during_decay 0.5 --corrupt_sd 0.2 --dataset mnist --decay_start 0.8 --encoder_layers 1000-500-250-250-250-10 --end_epoch 250 --epsilon 1.0 --id full_ladder_labeled-100 --initial_learning_rate 0.002 --logdir logs/ --lr_decay_frequency 10 --lrelu 0.0 --lrelu_a 0.1 --model ladder --num_labeled 100 --num_power_iters 3 --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --seed 370 --static_bn 0.99 --test_frequency_in_epochs 5 --ul_batch_size 250 --vadv_sd 0.5 --validation 0 --which_gpu 1 --xi 1e-6 