#!/usr/bin/env bash

python /home/sshinoda/tf-ssl/vat_ladder/vat_ladder.py --id "GaussVatCorr_seed-100" --seed 100 --vat_weight 0.0 --ent_weight 0.0 --which_gpu 1 --vat_corr --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10

python /home/sshinoda/tf-ssl/vat_ladder/vat_ladder.py --id "GaussVatCorr_seed-1111" --seed 1111 --vat_weight 0.0 --ent_weight 0.0 --which_gpu 1 --vat_corr --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10


