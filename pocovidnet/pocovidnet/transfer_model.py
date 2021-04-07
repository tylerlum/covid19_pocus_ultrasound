import tensorflow as tf
from keras.models import Model
from keras.layers import Dropout, Dense, TimeDistributed, Input, GlobalAveragePooling1D, Concatenate


def transferred_model_transformer(transferred_model_name, transferred_model_num_layers_remove,
                                  transferred_model_num_layers_add, trainable_base, nb_classes):
    print("WARNING: using transferred model, assuming transformer")
    from pocovidnet.transformer import TransformerBlock
    transferred_model = tf.keras.models.load_model(transferred_model_name, custom_objects={'TransformerBlock': TransformerBlock})

    # Remove old prediction layer
    model = Model(transferred_model.input, transferred_model.layers[-2].output)

    # Remove additional layers
    for i in range(transferred_model_num_layers_remove):
        model = Model(model.input, model.layers[-2].output)

    # Add additional hidden layers
    for i in range(transferred_model_num_layers_add):
        new_output = Dense(64, activation='relu')(model.output)
        new_output = Dropout(0.5)(new_output)
        model = Model(model.input, new_output)

    # Add new prediction layer
    model = Model(model.input, Dense(nb_classes, activation='softmax')(model.output))

    # Set trainable base
    if trainable_base:
        for layer in model.layers:
            layer.trainable = True

    return model


def transferred_model_average(transferred_model_name, transferred_model_num_layers_remove, transferred_model_num_layers_add, trainable_base, nb_classes):
    print("WARNING: using transferred model, assuming 2D_CNN_average")
    transferred_model = tf.keras.models.load_model(transferred_model_name)

    # Remove old average layer
    model = Model(transferred_model.input, transferred_model.layers[-2].output)

    # Remove old prediction layer
    model = Model(model.input, model.layers[-2].output)

    # Remove additional layers
    for i in range(transferred_model_num_layers_remove):
        model = Model(model.input, model.layers[-2].output)

    # Add additional hidden layers
    for i in range(transferred_model_num_layers_add):
        inp = Input(shape=(model.output.shape[2:]))
        x = Dense(64, activation='relu')(inp)
        x = Dropout(0.5)(x)
        end_of_cnn_model = Model(inputs=inp, outputs=x)
        model = Model(model.input, TimeDistributed(end_of_cnn_model)(model.output))

    # Add new prediction layer
    inp = Input(shape=(model.output.shape[2:]))
    x = Dense(nb_classes, activation='softmax')(inp)
    end_of_cnn_model = Model(inputs=inp, outputs=x)
    model = Model(model.input, TimeDistributed(end_of_cnn_model)(model.output))

    # Add new average layer
    model = Model(model.input, GlobalAveragePooling1D()(model.output))

    # Set trainable base
    if trainable_base:
        for layer in model.layers:
            layer.trainable = True
    return model


def two_transferred_models(transferred_model_name, transferred_model2_name, architecture, input_shape, transferred_model_num_layers_add, trainable_base, nb_classes):
    if "transformer" in architecture:
        print("WARNING: using 2 transferred models, assuming transformer")
        from pocovidnet.transformer import TransformerBlock
        transferred_model = tf.keras.models.load_model(transferred_model_name, custom_objects={'TransformerBlock': TransformerBlock})
        transferred_model2 = tf.keras.models.load_model(transferred_model2_name, custom_objects={'TransformerBlock': TransformerBlock})

        # Remove old prediction layer
        # Find embedding layer
        for layer in transferred_model.layers:
            if len(layer.output_shape) == 2 and not isinstance(layer, TransformerBlock):  # (batch_size, embed_dim)
                layerName = layer.name
                break
        for layer in transferred_model2.layers:
            if len(layer.output_shape) == 2 and not isinstance(layer, TransformerBlock):  # (batch_size, embed_dim)
                layerName2 = layer.name
                break

        inp = Input(shape=(input_shape))
        model = Model(transferred_model.input, transferred_model.get_layer(layerName).output)
        model2 = Model(transferred_model2.input, transferred_model2.get_layer(layerName2).output)
        x = model(inp)
        x2 = model2(inp)
        merged = Concatenate(axis=1)([x, x2])
        model = Model(inp, merged)

        # Add additional hidden layers
        for i in range(transferred_model_num_layers_add):
            new_output = Dense(256, activation='relu')(model.output)
            new_output = Dropout(0.5)(new_output)
            model = Model(model.input, new_output)

        # Add new prediction layer
        model = Model(model.input, Dense(nb_classes, activation='softmax')(model.output))

        # Set trainable base
        if trainable_base:
            for layer in model.layers:
                layer.trainable = True
    else:
        print("WARNING: using 2 transferred models, assuming 2D_CNN_average")
        transferred_model = tf.keras.models.load_model(transferred_model_name)
        transferred_model2 = tf.keras.models.load_model(transferred_model2_name)

        # Remove old average layer, prediction layer, and extra layer
        model = Model(transferred_model.input, transferred_model.layers[-4].output)
        model2 = Model(transferred_model2.input, transferred_model2.layers[-4].output)

        # Add additional hidden layers
        for i in range(transferred_model_num_layers_add):
            inp = Input(shape=(model.output.shape[2:]))
            x = Dense(64, activation='relu')(inp)
            x = Dropout(0.5)(x)
            end_of_cnn_model = Model(inputs=inp, outputs=x)
            model = Model(model.input, TimeDistributed(end_of_cnn_model)(model.output))

        # Add new prediction layer
        inp = Input(shape=(model.output.shape[2:]))
        x = Dense(nb_classes, activation='softmax')(inp)
        end_of_cnn_model = Model(inputs=inp, outputs=x)
        model = Model(model.input, TimeDistributed(end_of_cnn_model)(model.output))

        # Add new average layer
        model = Model(model.input, GlobalAveragePooling1D()(model.output))

        # Set trainable base
        if trainable_base:
            for layer in model.layers:
                layer.trainable = True

    return model
