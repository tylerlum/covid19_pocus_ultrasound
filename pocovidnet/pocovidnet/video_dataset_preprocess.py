from pocovidnet import PRETRAINED_CNN_PREPROCESS_FACTORY
import numpy as np


def preprocess_video_dataset(X, pretrained_cnn):
    # X : (n_samples, *dim, n_channels)
    preprocess_function = PRETRAINED_CNN_PREPROCESS_FACTORY[pretrained_cnn]
    preprocessed_x = [[preprocess_function(frame) for frame in video_clip] for video_clip in X]
    preprocessed_x = np.asarray(preprocessed_x)
    return preprocessed_x
