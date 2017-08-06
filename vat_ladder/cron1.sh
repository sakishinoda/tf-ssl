#!/usr/bin/env bash

source activate blaze
nohup python /home/sshinoda/tf-ssl/vat_ladder/vat_ladder.py --id "GaussVatRC_seed-1111" --seed 11 --vat_weight 1.0 --ent_weight 0.0 --which_gpu 0 --vat_rc --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 &
nohup python /home/sshinoda/tf-ssl/vat_ladder/vat_ladder.py --id "GaussVatRC_seed-1111" --seed 111 --vat_weight 1.0 --ent_weight 0.0 --which_gpu 1 --vat_rc --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 &

# GaussVatCorr
nohup python /home/sshinoda/tf-ssl/vat_ladder/vat_ladder.py --id "GaussVatCorr_seed-1" --description "VAT perturbation applied at each activation level to form the corrupted encoder. No overall VAT cost applied" --vat_weight 0.0 --ent_weight 0.0 --which_gpu 2 --vat_corr --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 &