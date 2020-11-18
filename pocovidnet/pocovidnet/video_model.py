
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation, Conv2D, Conv3D, Dense, Dropout, Flatten, MaxPooling2D, MaxPooling3D, Reshape, Lambda
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam


def get_video_model(input_shape, nb_classes):
    # Define model
    model = Sequential()
    model.add(
        Conv3D(
            16,
            kernel_size=(5, 3, 3),
            input_shape=(input_shape),
            # padding='same'
        )
    )
    model.add(Activation('relu'))
    model.add(Lambda(lambda x: tensorflow.keras.backend.squeeze(x, axis=1)))
    model.add(Conv2D(16, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model
