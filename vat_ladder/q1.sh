#!/usr/bin/env bash

# GPU 1

python mlp_ladder.py --rc_weights 0-0-0-0-0-0-0.5 --id "MlpGamma_seed-8340" --seed 8340 --test_frequency_in_epochs 5 --which_gpu 1
python mlp_ladder.py --rc_weights 0-0-0-0-0-0-0.5 --id "MlpGamma_seed-8794" --seed 8794 --test_frequency_in_epochs 5 --which_gpu 1
python mlp_ladder.py --rc_weights 0-0-0-0-0-0-0.5 --id "MlpGamma_seed-2773" --seed 2773 --test_frequency_in_epochs 5 --which_gpu 1
python mlp_ladder.py --rc_weights 0-0-0-0-0-0-0.5 --id "MlpGamma_seed-967" --seed 967 --test_frequency_in_epochs 5 --which_gpu 1
python mlp_ladder.py --rc_weights 0-0-0-0-0-0-0.5 --id "MlpGamma_seed-2368" --seed 2368 --test_frequency_in_epochs 5 --which_gpu 1

#python mlp_ladder.py --rc_weights 0-0-0-0-0-0-0.5 --id "MlpGamma_seed-1" --seed 1 --test_frequency_in_epochs 5 --which_gpu 1
#python mlp_ladder.py --rc_weights 0-0-0-0-0-0-0.5 --id "MlpGamma_seed-100" --seed 100 --test_frequency_in_epochs 5 --which_gpu 1
#python mlp_ladder.py --rc_weights 0-0-0-0-0-0-0.5 --id "MlpGamma_seed-11" --seed 11 --test_frequency_in_epochs 5 --which_gpu 1
#python mlp_ladder.py --rc_weights 0-0-0-0-0-0-0.5 --id "MlpGamma_seed-111" --seed 111 --test_frequency_in_epochs 5 --which_gpu 1
#python mlp_ladder.py --rc_weights 0-0-0-0-0-0-0.5 --id "MlpGamma_seed-1111" --seed 1111 --test_frequency_in_epochs 5 --which_gpu 1