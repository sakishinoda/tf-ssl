#!/usr/bin/env bash

# GPU 1

python vat_ladder.py --logdir logs/shallow/ --id shallow-nlw --model nlw --end_epoch 200 --decay_start 0.5 --which_gpu 1 --test_frequency_in_epochs 10 --encoder_layers 784-1200-1200-10 --rc_weights 1000-10-0.1-0.1 --epsilon 0.5-0.05-0.005-0.005 --lr_decay_frequency 10 --beta1 0.9 --beta1_during_decay 0.9  --num_power_iters 3

python vat_ladder.py --seed 1 --num_labeled 50 --batch_size 50 --ul_batch_size 250 --logdir logs/labeled-50/ --id lvan_labeled-50_seed-1 --model n --rc_weights 898.44421-8.81609-0.61101-0.11661-0.13746-0.50335-0.63461 --epsilon 0.03723 --end_epoch 200 --decay_start 0.5 --test_frequency_in_epochs 10 --lr_decay_frequency 10 --beta1 0.9 --beta1_during_decay 0.9 --which_gpu 1  --num_power_iters 3


