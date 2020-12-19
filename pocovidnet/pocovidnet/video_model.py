import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D, TimeDistributed, LSTM, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, Lambda, GlobalAveragePooling3D, Average, AveragePooling2D, ReLU, ZeroPadding3D
)
from tensorflow.keras.applications import VGG16, MobileNetV2, NASNetMobile
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from .utils import fix_layers


def get_video_model(input_shape, nb_classes, model_type="2D_CNN"):
    if model_type == "LSTM":
        return get_LSTM_model(input_shape, nb_classes)
    elif model_type == "3D_CNN":
        return get_3D_CNN_model(input_shape, nb_classes)
    elif model_type == "2+1D_CNN":
        return get_2plus1D_CNN_model(input_shape, nb_classes)
    elif model_type == "transformer":
        return get_transformer_model(input_shape, nb_classes)
    elif model_type == "2D_CNN":
        return get_2D_CNN_model(input_shape, nb_classes)
    elif model_type == "2stream":
        return get_2stream_model(input_shape, nb_classes)


def get_LSTM_model(input_shape, nb_classes):
    # Use pretrained vgg-model
    baseModel = VGG16(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(input_shape[1:]))
    )
    baseModel.trainable = False
    baseModel.summary()
    baseModelOut = GlobalAveragePooling2D()(baseModel.output)

    intermediate_model = Model(inputs=baseModel.input, outputs=baseModelOut)
    intermediate_model.summary()

    input_tensor = Input(shape=(input_shape))
    timeDistributed_layer = TimeDistributed(intermediate_model)(input_tensor)

    number_of_hidden_units = 64
    model = LSTM(number_of_hidden_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(timeDistributed_layer)
    model = LSTM(number_of_hidden_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)(model)
    model = Dense(2048, activation='relu')(model)
    model = Dense(128, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(nb_classes, activation='softmax')(model)
    model = Model(inputs=input_tensor, outputs=model)

    return model


def get_3D_CNN_model(input_shape, nb_classes):
    # Define model
    model = Sequential()
    model.add(
        Conv3D(
            32,
            kernel_size=(3, 3, 3),
            input_shape=(input_shape),
            padding='same'
        )
    )
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
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


def get_2plus1D_CNN_model(input_shape, nb_classes):
    # Define model
    model = Sequential()
    model.add(ZeroPadding3D(padding=(0, 1, 1), input_shape=(input_shape)))
    model.add(Conv3D(16, kernel_size=(1, 3, 3),))
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(1, 0, 0)))
    model.add(Conv3D(16, kernel_size=(3, 1, 1),))
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(Conv3D(16, kernel_size=(1, 3, 3),))
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(1, 0, 0)))
    model.add(Conv3D(16, kernel_size=(3, 1, 1),))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.5))

    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(Conv3D(16, kernel_size=(1, 3, 3),))
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(1, 0, 0)))
    model.add(Conv3D(16, kernel_size=(3, 1, 1),))
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(Conv3D(16, kernel_size=(1, 3, 3),))
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(1, 0, 0)))
    model.add(Conv3D(16, kernel_size=(3, 1, 1),))
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


def get_2D_CNN_model(input_shape, nb_classes):
    print(f"--------------input_shape = {input_shape}")
    baseModel = VGG16(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(input_shape[1:]))
    )

    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64)(headModel)
    headModel = BatchNormalization()(headModel)
    headModel = ReLU()(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(nb_classes, activation=tf.nn.softmax)(headModel)

    # place the head FC model on top of the base model
    cnn_model = Model(inputs=baseModel.input, outputs=headModel)

    trainable_layers = 1
    cnn_model = fix_layers(cnn_model, num_flex_layers=trainable_layers + 8)

    input_tensor = Input(shape=(input_shape))
    num_frames = input_shape[0]
    print(f"num_frames = {num_frames}")
    frame_predictions = []
    for frame_i in range(num_frames):
        frame = Lambda(lambda x: x[:, frame_i, :, :, :])(input_tensor)
        frame_prediction = cnn_model(frame)
        frame_predictions.append(frame_prediction)
    print(f"len(frame_predictions) = {len(frame_predictions)}")
    average = Average()(frame_predictions)
    return Model(inputs=input_tensor, outputs=average)


def get_transformer_model(input_shape, nb_classes):
    return None


def get_2stream_model(input_shape, nb_classes):
    return None
