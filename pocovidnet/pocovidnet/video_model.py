
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation, Conv2D, Conv3D, Dense, Dropout, Flatten, MaxPooling2D, MaxPooling3D, Reshape, Lambda, ZeroPadding3D
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam


def get_video_model(input_shape, nb_classes):
    # Define model
    model = Sequential()
    model.add(ZeroPadding3D(padding=(0, 1, 1), input_shape=(input_shape)))
    model.add(
        Conv3D(
            16,
            kernel_size=(1, 3, 3),
        )
    )
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(1, 0, 0)))
    model.add(
        Conv3D(
            16,
            kernel_size=(3, 1, 1),
        )
    )
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(
        Conv3D(
            16,
            kernel_size=(1, 3, 3),
        )
    )
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(1, 0, 0)))
    model.add(
        Conv3D(
            16,
            kernel_size=(3, 1, 1),
        )
    )
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.5))

    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(
        Conv3D(
            16,
            kernel_size=(1, 3, 3),
        )
    )
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(1, 0, 0)))
    model.add(
        Conv3D(
            16,
            kernel_size=(3, 1, 1),
        )
    )
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(
        Conv3D(
            16,
            kernel_size=(1, 3, 3),
        )
    )
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(1, 0, 0)))
    model.add(
        Conv3D(
            16,
            kernel_size=(3, 1, 1),
        )
    )
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(2048, activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model
