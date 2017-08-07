#!/usr/bin/env bash

# GPU 0

python vat_ladder.py --id "GaussVatCorrEntMin_seed-1" --seed 1 --vat_weight 0.0 --ent_weight 1.0 --which_gpu 0 --vat_corr --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10

python vat_ladder.py --id "GaussVatCorrEntMin_seed-100" --seed 100 --vat_weight 0.0 --ent_weight 1.0 --which_gpu 0 --vat_corr --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10

python vat_ladder.py --id "GaussVatCorrEntMin_seed-11" --seed 11 --vat_weight 0.0 --ent_weight 1.0 --which_gpu 0 --vat_corr --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10

python vat_ladder.py --id "GaussVatCorrEntMin_seed-111" --seed 111 --vat_weight 0.0 --ent_weight 1.0 --which_gpu 0 --vat_corr --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10

python vat_ladder.py --id "GaussVatCorrEntMin_seed-1111" --seed 1111 --vat_weight 0.0 --ent_weight 1.0 --which_gpu 0 --vat_corr --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10