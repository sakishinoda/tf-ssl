#!/usr/bin/env bash

# GPU 2
python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "LrDecayEveryEpoch_seed-111" --seed 111 --test_frequency_in_epochs 5 --which_gpu 2 --logdir "logs/MlpLadder/"
python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "LrDecayEveryEpoch_seed-1111" --seed 1111 --test_frequency_in_epochs 5 --which_gpu 2 --logdir "logs/MlpLadder/"

