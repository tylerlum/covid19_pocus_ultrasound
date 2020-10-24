#!/bin/bash
# Run from covid19_pocus_ultrasound/pocovidnet directory. Trains one model using Mobina's dataset. Saves to covid19_pocus_ultrasound/pocovidnet/models/mobina_dataset_model directory

# Test fold 0, vgg_base
python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/mobina_dataset' --validation_fold 1 --test_fold 2 --epochs 40 --batch_size 8 --model_name "mobina_dataset_model" --num_models 1
