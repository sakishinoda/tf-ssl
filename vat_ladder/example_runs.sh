#!/usr/bin/env bash


# Test ladder benchmark
echo "===> ladder <==="
python vat_ladder.py --logdir logs/test/ --id ladder --model ladder --do_not_save --end_epoch 1 --decay_start 1.0 --which_gpu 0 --test_frequency_in_epochs 1

# Test each model
echo "===> lvac <==="
python vat_ladder.py --logdir logs/test/ --id lvac --model c --do_not_save --end_epoch 1 --decay_start 1.0 --which_gpu 0 --test_frequency_in_epochs 1

echo "===> lvac-lw <==="
python vat_ladder.py --logdir logs/test/ --id lvac-lw --model clw --do_not_save --end_epoch 1 --decay_start 1.0 --which_gpu 0 --epsilon 1.0-0.1-0.01-0.01-0.01-0.01-0.01 --test_frequency_in_epochs 1

echo "===> lvan <==="
python vat_ladder.py --logdir logs/test/ --id lvan --model n --do_not_save --end_epoch 1 --decay_start 1.0 --which_gpu 0 --test_frequency_in_epochs 1

echo "===> lvan-lw <==="
python vat_ladder.py --logdir logs/test/ --id lvan#!/usr/bin/env bash



# Hyperopt
nohup python -u hyperopt.py --which_gpu 0 --num_labeled 1000 --dump c1000.res --model c --end_epoch 50 > c1000.log.out &

nohup python -u hyperopt.py --which_gpu 1 --num_labeled 100 --dump clw100.res --model clw --end_epoch 50 > clw100.log.out &

nohup python -u hyperopt.py --which_gpu 2 --num_labeled 100 --dump c100.res --model c --end_epoch 50 > c100.log.out &