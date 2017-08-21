#!/usr/bin/env bash

# GPU 2
python vat_ladder.py --logdir logs/smoothness/ --id lvan-lw --model nlw --end_epoch 200 --decay_start 0.5 --which_gpu 2 --test_frequency_in_epochs 10 --lr_decay_frequency 10 --beta1 0.9 --beta1_during_decay 0.9 --measure_smoothness --rc_weights 898.44421-8.81609-0.61101-0.11661-0.13746-0.50335-0.63461 --epsilon 0.11002-0.00930-0.00508-0.00001-0.00073-0.00113-0.00019

