#!/usr/bin/env bash

# GPU 0
python mlp_ladder.py --rc_weights 1000-10-0.1-0.1-0.1-0.1-0.1 --id "MlpLadder_cmb-amlp_seed-2368" --seed 2368 --test_frequency_in_epochs 5 --which_gpu 0 --logdir "logs/MlpLadderCmb/" --combinator "amlp"

