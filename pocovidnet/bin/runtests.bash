#!/bin/bash
# Run from covid19_pocus_ultrasound/pocovidnet directory. Trains 5 models using 4 folds for training and 1 fold for validation. Saves to covid19_pocus_ultrasound/pocovidnet/models/test directory

python3 scripts/train_covid19.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation' --fold 0 --epochs 40 --batch_size 8

python3 scripts/train_covid19.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation' --fold 1 --epochs 40 --batch_size 8

python3 scripts/train_covid19.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation' --fold 2 --epochs 40 --batch_size 8

python3 scripts/train_covid19.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation' --fold 3 --epochs 40 --batch_size 8

python3 scripts/train_covid19.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation' --fold 4 --epochs 40 --batch_size 8
