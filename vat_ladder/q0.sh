#!/usr/bin/env bash

# GPU 0

python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "DynamicBn_LrDecay-5_seed-1" --seed 1 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 0 --logdir "logs/MlpLadder/" --bn_decay "dynamic"
python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "DynamicBn_LrDecay-5_seed-100" --seed 100 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 0 --logdir "logs/MlpLadder/" --bn_decay "dynamic"
python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "DynamicBn_LrDecay-5_seed-11" --seed 11 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 0 --logdir "logs/MlpLadder/" --bn_decay "dynamic"
python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "DynamicBn_LrDecay-5_seed-111" --seed 111 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 0 --logdir "logs/MlpLadder/" --bn_decay "dynamic"
python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "DynamicBn_LrDecay-5_seed-1111" --seed 1111 --test_frequency_in_epochs 5 --lr_decay_frequency 5 --which_gpu 0 --logdir "logs/MlpLadder/" --bn_decay "dynamic"

