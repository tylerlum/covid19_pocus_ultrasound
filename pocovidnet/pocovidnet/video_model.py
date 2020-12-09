
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D, TimeDistributed, LSTM, Conv2D, MaxPooling2D
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam


def get_video_model(input_shape, nb_classes):
    # Define vgg-model
    model = Sequential()
    model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu'), input_shape=(input_shape)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2), padding='SAME',strides=(2,2))))
    model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2), padding='SAME',strides=(2,2))))
    model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2), padding='SAME',strides=(2,2))))
    model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2), padding='SAME',strides=(2,2))))
    model.add(TimeDistributed(Flatten()))

    # Define LSTM model
    number_of_hidden_units = 32
    model.add(LSTM(number_of_hidden_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(LSTM(number_of_hidden_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model
