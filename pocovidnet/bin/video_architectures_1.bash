#!/bin/bash
# Run from covid19_pocus_ultrasound/pocovidnet directory. Trains video classification models of different architectures

EPOCHS=50
FRAME_RATE=3
DEPTH=10
WANDB_PROJECT="covid-video-debugging"
RANDOM_SEED=150

# 2D_CNN_average
for i in {1..4}
do
  python scripts/video_classification.py --epoch $EPOCHS --model_id 2D_CNN_average --fr $FRAME_RATE --depth $DEPTH --wandb_project $WANDB_PROJECT --random_seed $RANDOM_SEED
done

# CNN_LSTM
for i in {1..4}
do
  python scripts/video_classification.py --epoch $EPOCHS --model_id CNN_LSTM --fr $FRAME_RATE --depth $DEPTH --wandb_project $WANDB_PROJECT --random_seed $RANDOM_SEED
done

# CNN_GRU
for i in {1..4}
do
  python scripts/video_classification.py --epoch $EPOCHS --model_id CNN_GRU --fr $FRAME_RATE --depth $DEPTH --wandb_project $WANDB_PROJECT --random_seed $RANDOM_SEED
done

# CNN_RNN
for i in {1..4}
do
  python scripts/video_classification.py --epoch $EPOCHS --model_id CNN_RNN --fr $FRAME_RATE --depth $DEPTH --wandb_project $WANDB_PROJECT --random_seed $RANDOM_SEED
done

# CNN_LSTM_integrated
for i in {1..4}
do
  python scripts/video_classification.py --epoch $EPOCHS --model_id CNN_LSTM_integrated --fr $FRAME_RATE --depth $DEPTH --wandb_project $WANDB_PROJECT --random_seed $RANDOM_SEED
done

# CNN_LSTM_bidirectional
for i in {1..4}
do
  python scripts/video_classification.py --epoch $EPOCHS --model_id CNN_LSTM_bidirectional --fr $FRAME_RATE --depth $DEPTH --wandb_project $WANDB_PROJECT --random_seed $RANDOM_SEED
done

# CNN_GRU_bidirectional
for i in {1..4}
do
  python scripts/video_classification.py --epoch $EPOCHS --model_id CNN_GRU_bidirectional --fr $FRAME_RATE --depth $DEPTH --wandb_project $WANDB_PROJECT --random_seed $RANDOM_SEED
done

# CNN_RNN_bidirectional
for i in {1..4}
do
  python scripts/video_classification.py --epoch $EPOCHS --model_id CNN_RNN_bidirectional --fr $FRAME_RATE --depth $DEPTH --wandb_project $WANDB_PROJECT --random_seed $RANDOM_SEED
done

# CNN_LSTM_integrated_bidirectional
for i in {1..4}
do
  python scripts/video_classification.py --epoch $EPOCHS --model_id CNN_LSTM_integrated_bidirectional --fr $FRAME_RATE --depth $DEPTH --wandb_project $WANDB_PROJECT --random_seed $RANDOM_SEED
done

# 3D_CNN
for i in {1..4}
do
  python scripts/video_classification.py --epoch $EPOCHS --model_id 3D_CNN --fr $FRAME_RATE --depth $DEPTH --wandb_project $WANDB_PROJECT --random_seed $RANDOM_SEED
done

# 2plus1D_CNN
for i in {1..4}
do
  python scripts/video_classification.py --epoch $EPOCHS --model_id 2plus1D_CNN --fr $FRAME_RATE --depth $DEPTH --wandb_project $WANDB_PROJECT --random_seed $RANDOM_SEED
done

# 2D_then_1D
for i in {1..4}
do
  python scripts/video_classification.py --epoch $EPOCHS --model_id 2D_then_1D --fr $FRAME_RATE --depth $DEPTH --wandb_project $WANDB_PROJECT --random_seed $RANDOM_SEED
done
