#!/bin/bash
# Run from covid19_pocus_ultrasound/pocovidnet directory. Evaluates uncertainty on hardcoded models.

python3 scripts/evaluate.py --data_dir ../data/cross_validation_balanced --model_dir models/ensemble_balanced/validation-fold-3_test-fold-4 --model_file last_epoch --fold 4 --mc_dropout True --test_time_augmentation True --deep_ensemble True
