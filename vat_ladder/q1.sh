#!/usr/bin/env bash

# GPU 1
python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "DynamicBn_seed-967" --seed 967 --test_frequency_in_epochs 5 --which_gpu 1 --logdir "logs/MlpLadder/" --bn_decay "dynamic"
python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "DynamicBn_seed-2368" --seed 2368 --test_frequency_in_epochs 5 --which_gpu 1 --logdir "logs/MlpLadder/" --bn_decay "dynamic"
