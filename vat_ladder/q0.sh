#!/usr/bin/env bash


# GPU 0
# Previously found best Deep MLP VAT

python vat_ladder.py  --batch_size 100 --beta1 0.9 --beta1_during_decay 0.5 --dataset mnist --decay_start 0.5 --do_not_save  --encoder_layers 784-1200-600-300-150-10 --end_epoch 400 --epsilon 5.0 --id DeepMlpVatMnist_seed-8340 --logdir logs/DeepMlpVatMnist/ --lr_decay_frequency 10 --model vat --seed 8340 --test_frequency_in_epochs 10 --ul_batch_size 250 --which_gpu 0