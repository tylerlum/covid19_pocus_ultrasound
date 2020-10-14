#!/bin/bash
# Run from covid19_pocus_ultrasound/pocovidnet directory. Trains 25 models using 3 folds for training, 1 fold for validation, and 1 fold for testing. Saves to covid19_pocus_ultrasound/pocovidnet/models/test_proper_base_and_cam directory

# Test fold 4, vgg_base
python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation' --validation_fold 0 --test_fold 4 --epochs 40 --batch_size 8 --model_name "test_proper_base2"

python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation' --validation_fold 1 --test_fold 4 --epochs 40 --batch_size 8 --model_name "test_proper_base2"

python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation' --validation_fold 2 --test_fold 4 --epochs 40 --batch_size 8 --model_name "test_proper_base2"

python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation' --validation_fold 3 --test_fold 4 --epochs 40 --batch_size 8 --model_name "test_proper_base2"


# Test fold 4, vgg_cam
python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_cam' --learning_rate 1e-4 --data_dir '../data/cross_validation' --validation_fold 0 --test_fold 4 --epochs 40 --batch_size 8 --model_name "test_proper_cam2"

python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_cam' --learning_rate 1e-4 --data_dir '../data/cross_validation' --validation_fold 1 --test_fold 4 --epochs 40 --batch_size 8 --model_name "test_proper_cam2"

python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_cam' --learning_rate 1e-4 --data_dir '../data/cross_validation' --validation_fold 2 --test_fold 4 --epochs 40 --batch_size 8 --model_name "test_proper_cam2"

python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_cam' --learning_rate 1e-4 --data_dir '../data/cross_validation' --validation_fold 3 --test_fold 4 --epochs 40 --batch_size 8 --model_name "test_proper_cam2"
