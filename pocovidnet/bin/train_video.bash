#!/bin/bash
# Run from covid19_pocus_ultrasound/pocovidnet directory. Trains one model using 3 folds for training, 1 fold for validation, and 1 fold for testing. Saves to covid19_pocus_ultrasound/pocovidnet/models/brightness_aug directory

# Test fold 4, vgg_base
python3 scripts/video_classification.py --epoch 40  --lr 1e-4
python3 scripts/video_classification.py --epoch 60  --lr 1e-4
python3 scripts/video_classification.py --epoch 40  --lr 1e-5
python3 scripts/video_classification.py --epoch 60  --lr 1e-5
python3 scripts/video_classification.py --epoch 40  --lr 1e-6
python3 scripts/video_classification.py --epoch 60  --lr 1e-6
