#!/bin/bash
# Run from covid19_pocus_ultrasound/pocovidnet directory. Trains many models using 3 folds for training, 1 fold for validation, and 1 fold for testing. Saves to covid19_pocus_ultrasound/pocovidnet/models/ensemble_balanced4 directory

for i in {1..2}
do
    python3 scripts/train_covid19_proper.py --trainable_base_layers 3 --model_id 'vgg_base' --learning_rate 1e-4 --data_dir '../data/cross_validation_balanced' --validation_fold 3 --test_fold 4 --epochs 3 --batch_size 8 --model_name "ensemble_balanced4_part${i}" --num_models 1
done

mkdir models/ensemble_balanced4/validation-fold-3_test-fold-4

# Copy over files
i=0
all_models=$(find ./models/ -wholename "*ensemble_balanced4_part*model-0")
for model in $all_models; do mv $model/ ./models/ensemble_balanced4/validation-fold-3_test-fold-4/model-$i; i=$(($i+1)); done

# Remove old folders
old_ones=$(find ./models/ -type d -name "*ensemble_balanced4_part*")
for f in $old_ones; do rm -r $f; done
