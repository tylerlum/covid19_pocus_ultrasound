#!/bin/bash
# Run from covid19_pocus_ultrasound/pocovidnet directory. Trains one model using 3 folds for training, 1 fold for validation, and 1 fold for testing. Saves to covid19_pocus_ultrasound/pocovidnet/models/brightness_aug directory

# Test fold 4, vgg_base
python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation_balanced' --validation_fold 4 --test_fold 3 --epochs 40 --batch_size 8 --model_name "brightness_aug" --num_models 2
python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation_balanced' --validation_fold 2 --test_fold 3 --epochs 40 --batch_size 8 --model_name "brightness_aug" --num_models 2
python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation_balanced' --validation_fold 1 --test_fold 3 --epochs 40 --batch_size 8 --model_name "brightness_aug" --num_models 2
python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation_balanced' --validation_fold 0 --test_fold 3 --epochs 40 --batch_size 8 --model_name "brightness_aug" --num_models 2
