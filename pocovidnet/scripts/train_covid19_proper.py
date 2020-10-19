import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imutils import paths
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from pocovidnet import MODEL_FACTORY
from pocovidnet.utils import Metrics, undersample, oversample

# Suppress logging
tf.get_logger().setLevel('ERROR')

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    '-d', '--data_dir', required=True, help='path to input dataset'
)
ap.add_argument('-m', '--model_dir', type=str, default='models/')
ap.add_argument(
    '-v', '--validation_fold', type=int, default='0', help='fold to take as validation data'
)
ap.add_argument(
    '-f', '--test_fold', type=int, default='1', help='fold to take as test data'
)
ap.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
ap.add_argument('-ep', '--epochs', type=int, default=20)
ap.add_argument('-bs', '--batch_size', type=int, default=16)
ap.add_argument('-t', '--trainable_base_layers', type=int, default=1)
ap.add_argument('-iw', '--img_width', type=int, default=224)
ap.add_argument('-ih', '--img_height', type=int, default=224)
ap.add_argument('-id', '--model_id', type=str, default='vgg_base')
ap.add_argument('-ls', '--log_softmax', type=bool, default=False)
ap.add_argument('-n', '--model_name', type=str, default='test')
ap.add_argument('-hs', '--hidden_size', type=int, default=64)
ap.add_argument('-nm', '--num_models', type=int, default=1)
args = vars(ap.parse_args())

# Initialize hyperparameters
DATA_DIR = args['data_dir']
MODEL_NAME = args['model_name']
VALIDATION_FOLD = args['validation_fold']
TEST_FOLD = args['test_fold']
MODEL_DIR = os.path.join(args['model_dir'], MODEL_NAME, f'validation-fold-{VALIDATION_FOLD}_test-fold-{TEST_FOLD}')
LR = args['learning_rate']
EPOCHS = args['epochs']
BATCH_SIZE = args['batch_size']
MODEL_ID = args['model_id']
TRAINABLE_BASE_LAYERS = args['trainable_base_layers']
IMG_WIDTH, IMG_HEIGHT = args['img_width'], args['img_height']
LOG_SOFTMAX = args['log_softmax']
HIDDEN_SIZE = args['hidden_size']
NUM_MODELS = args['num_models']

# Check if model class exists
if MODEL_ID not in MODEL_FACTORY.keys():
    raise ValueError(
        f'Model {MODEL_ID} not implemented. Choose from {MODEL_FACTORY.keys()}'
    )

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
print(f'Model parameters: {args}')
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print('Loading images...')
imagePaths = list(paths.list_images(DATA_DIR))
data = []
labels = []

print(f'selected validation-fold-{VALIDATION_FOLD}, test-fold-{TEST_FOLD}')

train_labels, validation_labels, test_labels = [], [], []
train_data, validation_data, test_data = [], [], []

# loop over folds
for imagePath in imagePaths:

    path_parts = imagePath.split(os.path.sep)
    # extract the split
    train_test = path_parts[-3][-1]
    # extract the class label from the filename
    label = path_parts[-2]
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # update the data and labels lists, respectively
    if train_test == str(TEST_FOLD):
        test_labels.append(label)
        test_data.append(image)
    elif train_test == str(VALIDATION_FOLD):
        validation_labels.append(label)
        validation_data.append(image)
    else:
        train_labels.append(label)
        train_data.append(image)

# Prepare data for model
print(
    f'\nNumber of training samples: {len(train_labels)} \n'
    f'\nNumber of validation samples: {len(validation_labels)} \n'
    f'Number of testing samples: {len(test_labels)}'
)

assert len(set(train_labels)) == len(set(test_labels)), (
    'Something went wrong. Some classes are only in train or test data.'
)  # yapf: disable

assert len(set(train_labels)) == len(set(validation_labels)), (
    'Something went wrong. Some classes are only in train or validation data.'
)  # yapf: disable

# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
train_data = np.array(train_data) / 255.0
validation_data = np.array(validation_data) / 255.0
test_data = np.array(test_data) / 255.0
train_labels_text = np.array(train_labels)
validation_labels_text = np.array(validation_labels)
test_labels_text = np.array(test_labels)

num_classes = len(set(train_labels))

# perform one-hot encoding on the labels
lb = LabelBinarizer()
lb.fit(train_labels_text)

train_labels = lb.transform(train_labels_text)
validation_labels = lb.transform(validation_labels_text)
test_labels = lb.transform(test_labels_text)

if num_classes == 2:
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)
    test_labels = to_categorical(test_labels, num_classes=num_classes)

trainX, trainY = train_data, train_labels
validationX, validationY = validation_data, validation_labels
testX, testY = test_data, test_labels

oversampledTrainX, oversampledTrainY = oversample(trainX, trainY, printText="training")
undersampledValidationX, undersampledValidationY = undersample(validationX, validationY, printText="validation")
undersampledTestX, undersampledTestY = undersample(testX, testY, printText="testing")
trainX, trainY = oversampledTrainX, oversampledTrainY
validationX, validationY = undersampledValidationX, undersampledValidationY
testX, testY = undersampledTestX, undersampledTestY

print('Class mappings are:', lb.classes_)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=10,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1
)

for i in range(NUM_MODELS):
    print(f"Training model {i}")
    print("====================================")
    this_model_dir = os.path.join(MODEL_DIR, f'model-{i}')
    if not os.path.exists(this_model_dir):
        os.makedirs(this_model_dir)


    # Load the VGG16 network
    model = MODEL_FACTORY[MODEL_ID](
        input_size=(IMG_WIDTH, IMG_HEIGHT, 3),
        num_classes=num_classes,
        trainable_layers=TRAINABLE_BASE_LAYERS,
        log_softmax=LOG_SOFTMAX,
        hidden_size=HIDDEN_SIZE
    )

    # Define callbacks
    earlyStopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    mcp_save = ModelCheckpoint(
        os.path.join(this_model_dir, 'epoch-{epoch:02d}'),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=7,
        verbose=1,
        epsilon=1e-4,
        mode='min'
    )
    # To show balanced accuracy
    metrics = Metrics((validationX, validationY), model)

    # compile model
    print('Compiling model...')
    opt = Adam(lr=LR, decay=LR / EPOCHS)
    loss = (
        tf.keras.losses.CategoricalCrossentropy() if not LOG_SOFTMAX else (
            lambda labels, targets: tf.reduce_mean(
                tf.reduce_sum(
                    -1 * tf.math.multiply(tf.cast(labels, tf.float32), targets),
                    axis=1
                )
            )
        )
    )

    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    print(f'Model has {model.count_params()} parameters')
    print(f'Model summary {model.summary()}')

    # train the head of the network
    print('Starting training model...')
    H = model.fit(
        trainAug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        batch_size=BATCH_SIZE,
        validation_data=(validationX, validationY),
        epochs=EPOCHS,
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss, metrics]
    )

    # make predictions on the testing set
    print('Evaluating network...')
    validationPredIdxs = model.predict(validationX, batch_size=BATCH_SIZE)
    testPredIdxs = model.predict(testX, batch_size=BATCH_SIZE)

    # CSV: save predictions for inspection:
    def savePredictionsToCSV(predIdxs, csvFilename, directory=this_model_dir):
        df = pd.DataFrame(predIdxs)
        df.to_csv(os.path.join(directory, csvFilename))
    savePredictionsToCSV(validationPredIdxs, "validation_preds_last_epoch.csv")
    savePredictionsToCSV(testPredIdxs, "test_preds_last_epoch.csv")

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    validationPredIdxs = np.argmax(validationPredIdxs, axis=1)
    testPredIdxs = np.argmax(testPredIdxs, axis=1)

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    def printAndSaveClassificationReport(y, predIdxs, classes, reportFilename, directory=this_model_dir):
        print(f'classification report sklearn for {reportFilename}')
        print(
            classification_report(
                y.argmax(axis=1), predIdxs, target_names=classes
            )
        )

        report = classification_report(
            y.argmax(axis=1), predIdxs, target_names=classes, output_dict=True
        )
        reportDf = pd.DataFrame(report).transpose()
        reportDf.to_csv(os.path.join(directory, reportFilename))

    printAndSaveClassificationReport(validationY, validationPredIdxs, lb.classes_, "validationReport.csv")
    printAndSaveClassificationReport(testY, testPredIdxs, lb.classes_, "testReport.csv")

    def printAndSaveConfusionMatrix(y, predIdxs, classes, confusionMatrixFilename, directory=this_model_dir):
        print(f'confusion matrix for {confusionMatrixFilename}')

        cm = confusion_matrix(y.argmax(axis=1), predIdxs)
        # show the confusion matrix, accuracy, sensitivity, and specificity
        print(cm)

        cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        cmDisplay.plot()
        plt.savefig(os.path.join(directory, confusionMatrixFilename))

    printAndSaveConfusionMatrix(validationY, validationPredIdxs, lb.classes_, "validationConfusionMatrix.png")
    printAndSaveConfusionMatrix(testY, testPredIdxs, lb.classes_, "testConfusionMatrix.png")

    # serialize the model to disk
    print(f'Saving COVID-19 detector model on {this_model_dir} data...')
    model.save(os.path.join(this_model_dir, 'last_epoch'), save_format='h5')

    # plot the training loss and accuracy
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, len(H.history['loss'])), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, len(H.history['val_loss'])), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, len(H.history['accuracy'])), H.history['accuracy'], label='train_acc')
    plt.plot(np.arange(0, len(H.history['val_accuracy'])), H.history['val_accuracy'], label='val_acc')
    plt.title('Training Loss and Accuracy on COVID-19 Dataset')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(this_model_dir, 'loss.png'))

print('Done, shuttting down!')
