#!/usr/bin/env bash


# GPU 0


# Test ladder benchmark
echo ladder
python mlp_ladder.py --logdir logs/test/ --id ladder --do_not_save --end_epoch 1 --decay_start 1.0 --which_gpu 0

# Test each model
echo lvac
python vat_ladder.py --logdir logs/test/ --id lvac --model c --do_not_save --end_epoch 1 --decay_start 1.0 --which_gpu 0

echo lvac-lw
python vat_ladder.py --logdir logs/test/ --id lvac-lw --model clw --do_not_save --end_epoch 1 --decay_start 1.0 --which_gpu 0 --epsilon 1.0-0.1-0.01-0.01-0.01-0.01-0.01

echo lvan
python vat_ladder.py --logdir logs/test/ --id lvan --model n --do_not_save --end_epoch 1 --decay_start 1.0 --which_gpu 0

echo lvan-lw
python vat_ladder.py --logdir logs/test/ --id lvan-lw --model nlw --do_not_save --end_epoch 1 --decay_start 1.0 --which_gpu 0 --epsilon 1.0-0.1-0.01-0.01-0.01-0.01-0.01
