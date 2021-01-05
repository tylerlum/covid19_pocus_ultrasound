import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D, TimeDistributed, LSTM, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, Lambda, GlobalAveragePooling3D, Average, AveragePooling2D, ReLU, ZeroPadding3D, Conv1D, GRU, ConvLSTM2D, Reshape, SimpleRNN, Bidirectional, LayerNormalization, Layer, GlobalAveragePooling1D
)
from tensorflow.keras.applications import VGG16, MobileNetV2, NASNetMobile
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from .utils import fix_layers
from pocovidnet.model import get_model
from pocovidnet.transformer import TransformerBlock
from tensorflow import keras


''' No information baseline '''
def get_baseline_model(input_shape, nb_classes):
    # Scales input by 0 right off the bat, so has no opportunity to improve
    vgg_model = get_model(input_size=input_shape[1:], log_softmax=False,)
    print("get_baseline_model")

    # Run vgg model on each frame
    input_tensor = Input(shape=(input_shape))
    zero_input = Lambda(lambda y: y*0)(input_tensor)

    num_frames = input_shape[0]
    if num_frames == 1:
        frame = Lambda(lambda x: x[:, 0, :, :, :])(zero_input)
        return Model(inputs=input_tensor, outputs=vgg_model(frame))

    else:
        frame_predictions = []
        for frame_i in range(num_frames):
            frame = Lambda(lambda x: x[:, frame_i, :, :, :])(zero_input)
            frame_prediction = vgg_model(frame)
            frame_predictions.append(frame_prediction)

        # Average activations
        average = Average()(frame_predictions)
        return Model(inputs=input_tensor, outputs=average)


''' Simple '''
def get_2D_CNN_average_model(input_shape, nb_classes):
    vgg_model = get_model(input_size=input_shape[1:], log_softmax=False,)

    # Run vgg model on each frame
    input_tensor = Input(shape=(input_shape))

    num_frames = input_shape[0]
    if num_frames == 1:
        frame = Lambda(lambda x: x[:, 0, :, :, :])(input_tensor)
        return Model(inputs=input_tensor, outputs=vgg_model(frame))

    else:
        frame_predictions = []
        for frame_i in range(num_frames):
            frame = Lambda(lambda x: x[:, frame_i, :, :, :])(input_tensor)
            frame_prediction = vgg_model(frame)
            frame_predictions.append(frame_prediction)

        # Average activations
        average = Average()(frame_predictions)
        return Model(inputs=input_tensor, outputs=average)


''' Recurrent '''
def get_CNN_LSTM_model(input_shape, nb_classes):
    return get_CNN_LSTM_model_helper(input_shape, nb_classes, bidirectional=False)


def get_CNN_GRU_model(input_shape, nb_classes):
    return get_CNN_GRU_model_helper(input_shape, nb_classes, bidirectional=False)


def get_CNN_RNN_model(input_shape, nb_classes):
    return get_CNN_RNN_model_helper(input_shape, nb_classes, bidirectional=False)


def get_CNN_LSTM_integrated_model(input_shape, nb_classes):
    return get_CNN_LSTM_integrated_model_helper(input_shape, nb_classes, bidirectional=False)


def get_CNN_LSTM_bidirectional_model(input_shape, nb_classes):
    return get_CNN_LSTM_model_helper(input_shape, nb_classes, bidirectional=True)


def get_CNN_GRU_bidirectional_model(input_shape, nb_classes):
    return get_CNN_GRU_model_helper(input_shape, nb_classes, bidirectional=True)


def get_CNN_RNN_bidirectional_model(input_shape, nb_classes):
    return get_CNN_RNN_model_helper(input_shape, nb_classes, bidirectional=True)


def get_CNN_LSTM_integrated_bidirectional_model(input_shape, nb_classes):
    return get_CNN_LSTM_integrated_model_helper(input_shape, nb_classes, bidirectional=True)


def get_CNN_LSTM_model_helper(input_shape, nb_classes, bidirectional):
    # Use pretrained vgg-model
    vgg_model = get_model(input_size=input_shape[1:], log_softmax=False,)

    # Remove the last activation+dropout layer for prediction
    vgg_model._layers.pop()
    vgg_model._layers.pop()
    vgg_model = Model(vgg_model.input, vgg_model._layers[-1].output)

    # Run LSTM over CNN outputs
    input_tensor = Input(shape=(input_shape))
    timeDistributed_layer = TimeDistributed(vgg_model)(input_tensor)

    number_of_hidden_units = 64
    if bidirectional:
        model = Bidirectional(LSTM(number_of_hidden_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(timeDistributed_layer)
    else:
        model = LSTM(number_of_hidden_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(timeDistributed_layer)
    if bidirectional:
        model = Bidirectional(LSTM(number_of_hidden_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))(model)
    else:
        model = LSTM(number_of_hidden_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)(model)
    model = Dense(2048, activation='relu')(model)
    model = Dense(128, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(nb_classes, activation='softmax')(model)
    model = Model(inputs=input_tensor, outputs=model)

    return model


def get_CNN_GRU_model_helper(input_shape, nb_classes, bidirectional):
    # Use pretrained vgg-model
    vgg_model = get_model(input_size=input_shape[1:], log_softmax=False,)

    # Remove the last activation+dropout layer for prediction
    vgg_model._layers.pop()
    vgg_model._layers.pop()
    vgg_model = Model(vgg_model.input, vgg_model._layers[-1].output)

    # Run GRU over CNN outputs
    input_tensor = Input(shape=(input_shape))
    timeDistributed_layer = TimeDistributed(vgg_model)(input_tensor)

    number_of_hidden_units = 64
    if bidirectional:
        model = Bidirectional(GRU(number_of_hidden_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(timeDistributed_layer)
    else:
        model = GRU(number_of_hidden_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(timeDistributed_layer)
    if bidirectional:
        model = Bidirectional(GRU(number_of_hidden_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))(model)
    else:
        model = GRU(number_of_hidden_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)(model)
    model = Dense(2048, activation='relu')(model)
    model = Dense(128, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(nb_classes, activation='softmax')(model)
    model = Model(inputs=input_tensor, outputs=model)

    return model


def get_CNN_RNN_model_helper(input_shape, nb_classes, bidirectional):
    # Use pretrained vgg-model
    vgg_model = get_model(input_size=input_shape[1:], log_softmax=False,)

    # Remove the last activation+dropout layer for prediction
    vgg_model._layers.pop()
    vgg_model._layers.pop()
    vgg_model = Model(vgg_model.input, vgg_model._layers[-1].output)

    # Run RNN over CNN outputs
    input_tensor = Input(shape=(input_shape))
    timeDistributed_layer = TimeDistributed(vgg_model)(input_tensor)

    number_of_hidden_units = 64
    if bidirectional:
        model = Bidirectional(SimpleRNN(number_of_hidden_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(timeDistributed_layer)
    else:
        model = SimpleRNN(number_of_hidden_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(timeDistributed_layer)
    if bidirectional:
        model = Bidirectional(SimpleRNN(number_of_hidden_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))(model)
    else:
        model = SimpleRNN(number_of_hidden_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)(model)
    model = Dense(2048, activation='relu')(model)
    model = Dense(128, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(nb_classes, activation='softmax')(model)
    model = Model(inputs=input_tensor, outputs=model)

    return model


def get_CNN_LSTM_integrated_model_helper(input_shape, nb_classes, bidirectional):
    # Use pretrained vgg-model
    vgg_model = get_model(input_size=input_shape[1:], log_softmax=False,)

    # Remove the layers after convolution
    vgg_model._layers.pop()
    vgg_model._layers.pop()
    vgg_model._layers.pop()
    vgg_model._layers.pop()
    vgg_model._layers.pop()
    vgg_model._layers.pop()
    vgg_model._layers.pop()
    vgg_model._layers.pop()
    vgg_model = Model(vgg_model.input, vgg_model._layers[-1].output)

    # Run GRU over CNN outputs
    input_tensor = Input(shape=(input_shape))
    timeDistributed_layer = TimeDistributed(vgg_model)(input_tensor)

    number_of_hidden_units = 32
    if bidirectional:
        model = Bidirectional(ConvLSTM2D(number_of_hidden_units, kernel_size=(3, 3), return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(timeDistributed_layer)
    else:
        model = ConvLSTM2D(number_of_hidden_units, kernel_size=(3, 3), return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(timeDistributed_layer)
    time_length = model.shape[1]
    model = Reshape((time_length, -1))(model)
    if bidirectional:
        model = Bidirectional(LSTM(number_of_hidden_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))(model)
    else:
        model = LSTM(number_of_hidden_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)(model)
    model = Dense(2048, activation='relu')(model)
    model = Dense(128, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(nb_classes, activation='softmax')(model)
    model = Model(inputs=input_tensor, outputs=model)

    return model


''' Convolutional '''
def get_3D_CNN_model(input_shape, nb_classes):
    # Define model
    model = Sequential()
    model.add(
        Conv3D(
            8,
            kernel_size=(3, 3, 3),
            input_shape=(input_shape),
            padding='same'
        )
    )
    model.add(Activation('relu'))
    model.add(Conv3D(8, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv3D(8, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(8, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model


def get_2plus1D_CNN_model(input_shape, nb_classes):
    # Define model
    model = Sequential()
    model.add(ZeroPadding3D(padding=(0, 1, 1), input_shape=(input_shape)))
    model.add(Conv3D(8, kernel_size=(1, 3, 3),))
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(1, 0, 0)))
    model.add(Conv3D(8, kernel_size=(3, 1, 1),))
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(Conv3D(8, kernel_size=(1, 3, 3),))
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(1, 0, 0)))
    model.add(Conv3D(8, kernel_size=(3, 1, 1),))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.5))

    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(Conv3D(8, kernel_size=(1, 3, 3),))
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(1, 0, 0)))
    model.add(Conv3D(8, kernel_size=(3, 1, 1),))
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(Conv3D(8, kernel_size=(1, 3, 3),))
    model.add(Activation('relu'))
    model.add(ZeroPadding3D(padding=(1, 0, 0)))
    model.add(Conv3D(8, kernel_size=(3, 1, 1),))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model


def get_2D_then_1D_model(input_shape, nb_classes):
    # Use pretrained vgg-model
    vgg_model = get_model(input_size=input_shape[1:], log_softmax=False,)

    # Remove the last activation+dropout layer for prediction
    vgg_model._layers.pop()
    vgg_model._layers.pop()
    vgg_model = Model(vgg_model.input, vgg_model._layers[-1].output)

    # Run Conv1D over CNN outputs
    input_tensor = Input(shape=(input_shape))
    timeDistributed_layer = TimeDistributed(vgg_model)(input_tensor)

    number_of_hidden_units = 64
    model = Conv1D(number_of_hidden_units, kernel_size=8, padding='same')(timeDistributed_layer)
    model = Conv1D(number_of_hidden_units, kernel_size=8, padding='same')(model)
    model = Flatten()(model)
    model = Dense(256, activation='relu')(model)
    model = Dense(64, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(nb_classes, activation='softmax')(model)
    model = Model(inputs=input_tensor, outputs=model)

    return model


''' Transformer '''
def get_CNN_transformer_model(input_shape, nb_classes):
    # Use pretrained vgg-model
    vgg_model = get_model(input_size=input_shape[1:], log_softmax=False,)

    # Remove the last activation+dropout layer for prediction
    vgg_model._layers.pop()
    vgg_model._layers.pop()
    vgg_model = Model(vgg_model.input, vgg_model._layers[-1].output)

    # Run Conv1D over CNN outputs
    input_tensor = Input(shape=(input_shape))
    timeDistributed_layer = TimeDistributed(vgg_model)(input_tensor)

    # timeDistributed_layer.shape = (batch_size, timesteps, embed_dim)
    timesteps = timeDistributed_layer.shape[1]
    embed_dim = timeDistributed_layer.shape[2]
    num_heads = 4  # Requres embed_dim % num_heads == 0
    number_of_hidden_units = 64
    transformer_block1 = TransformerBlock(embed_dim, num_heads, number_of_hidden_units, timesteps)
    transformer_block2 = TransformerBlock(embed_dim, num_heads, number_of_hidden_units, timesteps)
    model = transformer_block1(timeDistributed_layer)
    model = transformer_block2(model)
    model = GlobalAveragePooling1D()(model)
    model = Dense(256, activation='relu')(model)
    model = Dense(64, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(nb_classes, activation='softmax')(model)
    model = Model(inputs=input_tensor, outputs=model)

    return model


''' Model Genesis '''
def get_model_genesis_model(input_shape, nb_classes):
    return None


''' Two stream optical flow '''
def get_2stream_model(input_shape, nb_classes):
    return None


''' CVPR '''
def get_gate_shift_model(input_shape, nb_classes):
    return None


def get_tea_model(input_shape, nb_classes):
    return None
