#!/bin/bash
# Run from covid19_pocus_ultrasound/pocovidnet directory. Trains 25 models using 3 folds for training, 1 fold for validation, and 1 fold for testing. Saves to covid19_pocus_ultrasound/pocovidnet/models/test_proper_base_and_cam directory

# Test fold 1, vgg_base
python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation10' --validation_fold 0 --test_fold 1 --epochs 40 --batch_size 8 --model_name "test_proper_base4"

python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation10' --validation_fold 2 --test_fold 1 --epochs 40 --batch_size 8 --model_name "test_proper_base4"

python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation10' --validation_fold 3 --test_fold 1 --epochs 40 --batch_size 8 --model_name "test_proper_base4"

python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation10' --validation_fold 4 --test_fold 1 --epochs 40 --batch_size 8 --model_name "test_proper_base4"

# Test fold 1, vgg_cam
python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_cam' --learning_rate 1e-4 --data_dir '../data/cross_validation10' --validation_fold 0 --test_fold 1 --epochs 40 --batch_size 8 --model_name "test_proper_cam4"

python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_cam' --learning_rate 1e-4 --data_dir '../data/cross_validation10' --validation_fold 2 --test_fold 1 --epochs 40 --batch_size 8 --model_name "test_proper_cam4"

python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_cam' --learning_rate 1e-4 --data_dir '../data/cross_validation10' --validation_fold 3 --test_fold 1 --epochs 40 --batch_size 8 --model_name "test_proper_cam4"

python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_cam' --learning_rate 1e-4 --data_dir '../data/cross_validation10' --validation_fold 4 --test_fold 1 --epochs 40 --batch_size 8 --model_name "test_proper_cam4"
