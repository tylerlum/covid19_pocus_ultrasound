
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam


def get_video_model(input_shape, nb_classes, batch_norm=False):
    # Define model
    model = Sequential()
    model.add(
        Conv3D(
            32,
            kernel_size=(3, 3, 3),
            input_shape=(input_shape),
            padding='same',
            activation='relu',
        )
    )
    if batch_norm: model.add(BatchNormalization())
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same', activation='softmax'))
    if batch_norm: model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    if batch_norm: model.add(BatchNormalization())
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same', activation='softmax'))
    if batch_norm: model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(2048, activation=None))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model
