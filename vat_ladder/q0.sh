#!/usr/bin/env bash

# GPU 0

python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "LrDecayEveryEpoch_seed-8340" --seed 8340 --test_frequency_in_epochs 5 --which_gpu 0 --logdir "logs/MlpLadder/" --combinator "gauss" --bn_decay "constant"
python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "LrDecayEveryEpoch_seed-8794" --seed 8794 --test_frequency_in_epochs 5 --which_gpu 0 --logdir "logs/MlpLadder/" --combinator "gauss" --bn_decay "constant"
python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "LrDecayEveryEpoch_seed-2773" --seed 2773 --test_frequency_in_epochs 5 --which_gpu 0 --logdir "logs/MlpLadder/" --combinator "gauss" --bn_decay "constant"
python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "LrDecayEveryEpoch_seed-967" --seed 967 --test_frequency_in_epochs 5 --which_gpu 0 --logdir "logs/MlpLadder/" --combinator "gauss" --bn_decay "constant"
python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "LrDecayEveryEpoch_seed-2368" --seed 2368 --test_frequency_in_epochs 5 --which_gpu 0 --logdir "logs/MlpLadder/" --combinator "gauss" --bn_decay "constant"