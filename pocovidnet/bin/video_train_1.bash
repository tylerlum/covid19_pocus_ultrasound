#!/bin/bash
# Run from covid19_pocus_ultrasound/pocovidnet directory. Trains video classification models

# 2D_CNN_average no augment
python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --augment False --load True
python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --augment False --load True
python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --augment False --load True
python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --augment False --load True
python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --augment False --load True

# 2D_CNN_average augment
python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --augment True --load True
python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --augment True --load True
python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --augment True --load True
python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --augment True --load True
python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --augment True --load True

# CNN_LSTM no augment
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --augment False --load True
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --augment False --load True
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --augment False --load True
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --augment False --load True
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --augment False --load True

# CNN_LSTM augment
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --augment True --load True
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --augment True --load True
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --augment True --load True
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --augment True --load True
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --augment True --load True
