#!/usr/bin/env bash


# GPU 0
python vat_ladder.py --logdir logs/smoothness/ --id ladder --model ladder  --end_epoch 200 --decay_start 0.5 --test_frequency_in_epochs 10 --lr_decay_frequency 10 --beta1 0.9 --beta1_during_decay 0.9 --measure_smoothness --which_gpu 0