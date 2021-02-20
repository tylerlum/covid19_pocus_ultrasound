import numpy as np
import keras
import imgaug.augmenters as iaa
import cv2
from .video_dataset_preprocess import preprocess_video_dataset

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, Y, batch_size, dim, shuffle, pretrained_cnn):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.Y = Y
        self.X = X
        self.shuffle = shuffle
        self.pretrained_cnn = pretrained_cnn
        self.on_epoch_end()
        self.augmentation = iaa.Sequential([
            iaa.Multiply((0.5, 1.5)),
            iaa.Add((-50, 50)),
            iaa.Fliplr(0.5),
            iaa.Affine(scale=(0.9, 1.1)),
            iaa.Affine(rotate=(-5, 5)),
            iaa.TranslateX(percent=(-0.1, 0.1)),
            iaa.TranslateY(percent=(-0.1, 0.1)),
            iaa.Crop(percent=(0, 0.1)),
            iaa.Pad(percent=(0, 0.1)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.GammaContrast((0.5, 2.0)),
            ])

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Get batch
        X_temp = [self.X[k] for k in indexes]
        Y_temp = [self.Y[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(X_temp, Y_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, X_temp, y_list):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x_list = []

        # For each video, use same augmentation for all frames
        for x in X_temp:
            deterministic = self.augmentation.to_deterministic()
            augmented_x = [deterministic.augment_image(frame) for frame in x]
            x_list.append(augmented_x)

        X = np.array(x_list)
        X = preprocess_video_dataset(X, self.pretrained_cnn)
        Y = np.array(y_list)

        return X, Y
