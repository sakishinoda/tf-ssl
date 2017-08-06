#!/usr/bin/env bash

# Gauss is also Gauss_seed-1
# nohup python vat_ladder.py --id "Gauss" --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --vat_weight 0.0 --description "Standard Gauss combinator, no VAT cost" --which_gpu 0 &
# nohup python vat_ladder.py --id "Gauss_Alpha-1.0" --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --vat_weight 1.0 --description "Standard Gauss combinator, VAT cost 1.0"  --which_gpu 1 &
# nohup python vat_ladder.py --id "Gauss_Alpha-10.0" --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --vat_weight 10.0 --description "Standard Gauss combinator, VAT cost 10.0"  --which_gpu 2 &

# nohup python vat_ladder.py --id "Gauss_seed-10" --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --vat_weight 0.0 --description "Standard Gauss combinator, no VAT cost" --which_gpu 0 --seed 10 &
# nohup python vat_ladder.py --id "Gauss_seed-100" --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --vat_weight 0.0 --description "Standard Gauss combinator, no VAT cost" --which_gpu 1 --seed 100 &
#nohup python vat_ladder.py --id "Gauss_seed-1000" --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --vat_weight 0.0 --description "Standard Gauss combinator, no VAT cost" --which_gpu 0 --seed 1000 &

nohup python vat_ladder.py --id "GaussVatEntMin_seed-100" --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --vat_weight 1.0 --ent_weight 1.0 --which_gpu 0 --seed 100 &
nohup python vat_ladder.py --id "GaussVatEntMin_seed-11" --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --vat_weight 1.0 --ent_weight 1.0 --which_gpu 1 --seed 11 &
nohup python vat_ladder.py --id "GaussVatEntMin_seed-111" --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --vat_weight 1.0 --ent_weight 1.0 --which_gpu 2 --seed 111 &