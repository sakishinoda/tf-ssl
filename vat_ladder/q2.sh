#!/usr/bin/env bash

# GPU 2
python vat_ladder.py --id "CorrGaussVat_seed-1" --seed 1 --vat_weight 1.0 --ent_weight 1.0 --which_gpu 2 --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --corrupt "vatgauss"

python vat_ladder.py --id "CorrGaussVat_seed-11" --seed 11 --vat_weight 1.0 --ent_weight 1.0 --which_gpu 2 --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --corrupt "vatgauss"

python vat_ladder.py --id "CorrGaussVat_seed-111" --seed 111 --vat_weight 1.0 --ent_weight 1.0 --which_gpu 2 --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --corrupt "vatgauss"

python vat_ladder.py --id "CorrGaussVat_seed-1111" --seed 1111 --vat_weight 1.0 --ent_weight 1.0 --which_gpu 2 --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --corrupt "vatgauss"

python vat_ladder.py --id "CorrGaussVat_seed-100" --seed 100 --vat_weight 1.0 --ent_weight 1.0 --which_gpu 2 --rc_weights 1000.0-10.0-0.10-0.10-0.10-0.10-0.10 --corrupt "vatgauss"
