#!/usr/bin/env bash

# GPU 2
python vat_ladder_new.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --vat_weight 0.1 --id "vat-0.1_seed-8340" --seed 8340 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 2 --logdir "logs/VLadder/" --bn_decay "constant"
python vat_ladder_new.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --vat_weight 0.1 --id "vat-0.1_seed-8794" --seed 8794 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 2 --logdir "logs/VLadder/" --bn_decay "constant"
python vat_ladder_new.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --vat_weight 0.1 --id "vat-0.1_seed-2773" --seed 2773 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 2 --logdir "logs/VLadder/" --bn_decay "constant"
python vat_ladder_new.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --vat_weight 0.1 --id "vat-0.1_seed-967" --seed 967 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 2 --logdir "logs/VLadder/" --bn_decay "constant"
python vat_ladder_new.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --vat_weight 0.1 --id "vat-0.1_seed-2368" --seed 2368 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 2 --logdir "logs/VLadder/" --bn_decay "constant"