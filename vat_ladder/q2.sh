#!/usr/bin/env bash

# GPU 2
# Benchmark plain MLP VAT again
# Minor difference in number of labeled/unlabeled

#python vat_ladder.py  --batch_size 100 --beta1 0.5 --beta1_during_decay 0.5 --dataset mnist --decay_start 0.5 --do_not_save  --encoder_layers 784-1200-600-300-150-10 --end_epoch 400 --epsilon 5.0 --id DeepMlpVatMnist_seed-1 --logdir logs/DeepMlpVatMnist/ --lr_decay_frequency 10 --model vat --seed 1 --static_bn 0.99 --test_frequency_in_epochs 10 --ul_batch_size 250 --vadv_sd 0.5 --which_gpu 2
#
#python vat_ladder.py  --batch_size 100 --beta1 0.5 --beta1_during_decay 0.5 --dataset mnist --decay_start 0.5 --do_not_save  --encoder_layers 784-1200-600-300-150-10 --end_epoch 400 --epsilon 5.0 --id DeepMlpVatMnist_seed-11 --logdir logs/DeepMlpVatMnist/ --lr_decay_frequency 10 --model vat --seed 11 --static_bn 0.99 --test_frequency_in_epochs 10 --ul_batch_size 250 --vadv_sd 0.5 --which_gpu 2
#
#python vat_ladder.py  --batch_size 100 --beta1 0.5 --beta1_during_decay 0.5 --dataset mnist --decay_start 0.5 --do_not_save  --encoder_layers 784-1200-600-300-150-10 --end_epoch 400 --epsilon 5.0 --id DeepMlpVatMnist_seed-100 --logdir logs/DeepMlpVatMnist/ --lr_decay_frequency 10 --model vat --seed 100 --static_bn 0.99 --test_frequency_in_epochs 10 --ul_batch_size 250 --vadv_sd 0.5 --which_gpu 2

python vat_ladder.py  --batch_size 64 --beta1 0.5 --beta1_during_decay 0.5 --dataset mnist --decay_start 0.5 --do_not_save  --encoder_layers 784-1200-600-300-150-10 --end_epoch 400 --epsilon 5.0 --id DeepMlpVatMnist_seed-8340 --logdir logs/DeepMlpVatMnist/ --lr_decay_frequency 1 --model vat --seed 8340 --static_bn 0.99 --test_frequency_in_epochs 1 --ul_batch_size 256 --vadv_sd 0.5 --which_gpu 2 --lrelu_a 0.1