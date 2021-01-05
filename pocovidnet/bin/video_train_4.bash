#!/bin/bash
# Run from covid19_pocus_ultrasound/pocovidnet directory. Trains video classification models

############### DEPTH 1 ###############
# 2D_CNN_average no augment depth 1
for i in {1..2}
do
  python scripts/video_classification.py --epoch 1 --model_id 2D_CNN_average --fr 3 --depth 1
done

# 2D_CNN_average augment depth 1
for i in {1..2}
do
python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 1 --augment
done

# CNN_LSTM no augment depth 1
for i in {1..2}
do
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 1
done

# CNN_LSTM augment depth 1
for i in {1..2}
do
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 1 --augment
done

############### DEPTH 5 ###############
# 2D_CNN_average no augment depth 5
for i in {1..2}
do
  python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5
done

# 2D_CNN_average augment depth 5
for i in {1..2}
do
python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --augment
done

# CNN_LSTM no augment depth 5
for i in {1..2}
do
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5
done

# CNN_LSTM augment depth 5
for i in {1..2}
do
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --augment
done

############### DEPTH 10 ###############
# 2D_CNN_average no augment depth 10
for i in {1..2}
do
  python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 10
done

# 2D_CNN_average augment depth 10
for i in {1..2}
do
python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 10 --augment
done

# CNN_LSTM no augment depth 10
for i in {1..2}
do
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 10
done

# CNN_LSTM augment depth 10
for i in {1..2}
do
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 10 --augment
done

############### DEPTH 15 ###############
# 2D_CNN_average no augment depth 15
for i in {1..2}
do
  python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 15
done

# 2D_CNN_average augment depth 15
for i in {1..2}
do
python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 15 --augment
done

# CNN_LSTM no augment depth 15
for i in {1..2}
do
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 15
done

# CNN_LSTM augment depth 15
for i in {1..2}
do
python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 15 --augment
done
