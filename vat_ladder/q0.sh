#!/usr/bin/env bash

# GPU 0

python /home/sshinoda/tf-ssl/vat_ladder/vat_ladder.py --id "GaussVatCorr_seed-1" --seed 1 --vat_weight 0.0 --ent_weight 0.0 --which_gpu 0 --vat_corr --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --test_frequency_in_epochs 5

python /home/sshinoda/tf-ssl/vat_ladder/vat_ladder.py --id "GaussVatCorr_seed-11" --seed 11 --vat_weight 0.0 --ent_weight 0.0 --which_gpu 0 --vat_corr --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --test_frequency_in_epochs 5

python /home/sshinoda/tf-ssl/vat_ladder/vat_ladder.py --id "GaussVatCorr_seed-100" --seed 100 --vat_weight 0.0 --ent_weight 0.0 --which_gpu 0 --vat_corr --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --test_frequency_in_epochs 5


