#!/usr/bin/env bash

# GPU 1
python vat_ladder_new.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --vat_weight 1.0 --id "vat-1.0_seed-8340" --seed 8340 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 1 --logdir "logs/VLadder/" --bn_decay "constant"
python vat_ladder_new.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --vat_weight 1.0 --id "vat-1.0_seed-8794" --seed 8794 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 1 --logdir "logs/VLadder/" --bn_decay "constant"
python vat_ladder_new.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --vat_weight 1.0 --id "vat-1.0_seed-2773" --seed 2773 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 1 --logdir "logs/VLadder/" --bn_decay "constant"
python vat_ladder_new.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --vat_weight 1.0 --id "vat-1.0_seed-967" --seed 967 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 1 --logdir "logs/VLadder/" --bn_decay "constant"
python vat_ladder_new.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --vat_weight 1.0 --id "vat-1.0_seed-2368" --seed 2368 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 1 --logdir "logs/VLadder/" --bn_decay "constant"