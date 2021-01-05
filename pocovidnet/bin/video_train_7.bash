#!/bin/bash
# Run from covid19_pocus_ultrasound/pocovidnet directory. Trains video classification models

############### DEPTH 1 ###############
# 2D_CNN_average no augment depth 1
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 1 --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 1 --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

# 2D_CNN_average augment depth 1
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 1 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 1 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

# CNN_LSTM no augment depth 1
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 1 --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 1 --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

# CNN_LSTM augment depth 1
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 1 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 1 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

############### DEPTH 5 ###############
# 2D_CNN_average no augment depth 5
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

# 2D_CNN_average augment depth 5
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 5 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

# CNN_LSTM no augment depth 5
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

# CNN_LSTM augment depth 5
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 5 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

############### DEPTH 10 ###############
# 2D_CNN_average no augment depth 10
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 10 --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 10 --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

# 2D_CNN_average augment depth 10
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 10 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 10 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

# CNN_LSTM no augment depth 10
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 10 --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 10 --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

# CNN_LSTM augment depth 10
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 10 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 10 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

############### DEPTH 15 ###############
# 2D_CNN_average no augment depth 15
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 15 --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 15 --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

# 2D_CNN_average augment depth 15
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 15 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id 2D_CNN_average --fr 3 --depth 15 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

# CNN_LSTM no augment depth 15
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 15 --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 15 --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done

# CNN_LSTM augment depth 15
for i in {1..4}
do
  if [ $i -gt 2 ]; then
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 15 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120
  else
    python scripts/video_classification.py --epoch 50 --model_id CNN_LSTM --fr 3 --depth 15 --augment --wandb_project covid-video-overnight-2-repeat --random_seed 120 --reduce_lr
  fi
done
