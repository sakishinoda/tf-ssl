#!/usr/bin/env bash

# GPU 1
python vat_ladder.py --logdir logs/smoothness/ --id lvac --model c --end_epoch 200 --decay_start 0.5 --which_gpu 1 --test_frequency_in_epochs 10 --lr_decay_frequency 10 --beta1 0.9 --beta1_during_decay 0.9 --measure_smoothness --rc_weights 898.44421-20.73306-0.17875-0.31394-0.02214-0.39981-0.04065 --epsilon 0.03723
