import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import cv2
import argparse
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Dropout
from keras.models import Model, Input
from keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import numpy as np
from pocovidnet.utils import undersample
from scripts.evaluate_uncertainty import create_mc_model, accuracy
from datetime import datetime
from datetime import date

def save_evaluation_files(labels, logits, classes, start_of_filename, directory):
    # CSV: save predictions for inspection:
    def savePredictionsToCSV(logits, start_of_filename, directory):
        df = pd.DataFrame(logits)
        df.to_csv(os.path.join(directory, start_of_filename + "preds.csv"))

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    def printAndSaveClassificationReport(y, predIdxs, classes, start_of_filename, directory):
        print(
            classification_report(
                y.argmax(axis=1), predIdxs, target_names=classes
            )
        )

        report = classification_report(
            y.argmax(axis=1), predIdxs, target_names=classes, output_dict=True
        )
        reportDf = pd.DataFrame(report).transpose()
        reportDf.to_csv(os.path.join(directory, start_of_filename + "Report.csv"))

    def printAndSaveConfusionMatrix(y, predIdxs, classes, start_of_filename, directory):
        cm = confusion_matrix(y.argmax(axis=1), predIdxs)
        # show the confusion matrix, accuracy, sensitivity, and specificity
        print(cm)

        cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        cmDisplay.plot()
        plt.savefig(os.path.join(directory, start_of_filename + "ConfusionMatrix.png"))

    # make predictions on the testing set
    savePredictionsToCSV(logits, start_of_filename, directory)

    predIdxs = np.argmax(logits, axis=1)
    printAndSaveClassificationReport(labels, predIdxs, classes, start_of_filename, directory)
    printAndSaveConfusionMatrix(labels, predIdxs, classes, start_of_filename, directory)


def get_dataset():
    # Get data
    data, labels = [], []

    # loop over folds
    imagePaths = list(paths.list_images(DATA_DIR))
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
        if train_test == str(FOLD):
            labels.append(label)
            data.append(image)

    # Prepare data for model
    print(
        f'\nNumber of datapoints: {len(data)} \n'
    )
    print(
        f'\nNumber of labels: {len(labels)} \n'
    )

    # convert the data and labels to NumPy arrays while scaling the pixel
    # intensities to the range [0, 255]
    data = np.array(data) / 255.0
    labels_text = np.array(labels)

    num_classes = len(set(labels))

    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    lb.fit(labels_text)

    labels = lb.transform(labels_text)

    if num_classes == 2:
        labels = to_categorical(labels, num_classes=num_classes)

    print('Class mappings are:', lb.classes_)
    data, labels = undersample(data, labels, printText="testing")

    return data, labels, lb.classes_


if __name__ == "__main__":
    IMG_WIDTH, IMG_HEIGHT = 224, 224

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-d', '--data_dir', required=True, help='path to input dataset'
    )
    ap.add_argument('-m', '--model_dir', required=True, type=str)
    ap.add_argument('-n', '--model_file', required=True, type=str)
    ap.add_argument(
        '-f', '--fold', required=True, type=int, help='evaluate on this fold'
    )
    ap.add_argument('-o', '--output_dir', type=str, default='outputs/')
    ap.add_argument('-a', '--mc_dropout', type=bool, default=False)
    ap.add_argument('-b', '--test_time_augmentation', type=bool, default=False)
    ap.add_argument('-c', '--deep_ensemble', type=bool, default=False)
    args = vars(ap.parse_args())

    # Initialize hyperparameters
    DATA_DIR = args['data_dir']
    MODEL_DIR = args['model_dir']
    MODEL_FILE = args['model_file']
    FOLD = args['fold']
    OUTPUT_DIR = args['output_dir']
    MC_DROPOUT = args['mc_dropout']
    TEST_TIME_AUGMENTATION = args['test_time_augmentation']
    DEEP_ENSEMBLE = args['deep_ensemble']
    REGULAR = True

    print(f'Evaluating with: {args}')

    # Setup output dir
    datestring = date.today().strftime("%b-%d-%Y") + "_" + datetime.now().strftime('%H-%M-%S')
    FINAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f"{__file__}__Data-{DATA_DIR}__Model-{MODEL_DIR}__Fold-{FOLD}__MC-{MC_DROPOUT}__TTA-{TEST_TIME_AUGMENTATION}__ENSEMBLE-{DEEP_ENSEMBLE}__".replace('/', '|') + datestring)
    if not os.path.exists(FINAL_OUTPUT_DIR):
        os.makedirs(FINAL_OUTPUT_DIR)

    # Get dataset
    data, labels, classes = get_dataset()

    # Consider shrinking dataset for easier understanding
    # start_idx, end_idx = 10, 14
    # data, labels = np.array(data[start_idx:end_idx]), np.array(labels[start_idx:end_idx])


    if REGULAR:
        print(f"Running REGULAR model")
        print("========================")

        # Create MC model
        model_path = os.path.join(MODEL_DIR, "model-0", MODEL_FILE)
        print(f"Looking for model at {model_path}")
        model = tf.keras.models.load_model(model_path)

        # make predictions on the testing set
        logits = model.predict(data)
        save_evaluation_files(labels, logits, classes, "regular", FINAL_OUTPUT_DIR)

        import shap
        explainer = shap.GradientExplainer(model, data)
        num_examples = 20
        shap_values = explainer.shap_values(data[:num_examples])
        print(len(shap_values))
        print(len(shap_values[0]))
        shap.image_plot([shap_values[i] for i in range(len(shap_values))], data[:num_examples])
        plt.savefig(os.path.join(FINAL_OUTPUT_DIR, "shap.png"))


    # Setup model
    if MC_DROPOUT:
        NUM_MC_DROPOUT_RUNS = 20
        print(f"Running {NUM_MC_DROPOUT_RUNS} runs of MC Dropout")
        print("========================")

        # Create MC model
        model_path = os.path.join(MODEL_DIR, "model-0", MODEL_FILE)
        print(f"Looking for model at {model_path}")
        model = tf.keras.models.load_model(model_path)
        mc_model = create_mc_model(model)

        # Compute logits
        all_logits = np.zeros((NUM_MC_DROPOUT_RUNS, labels.shape[0], labels.shape[1]))
        accuracies = []
        for i in range(NUM_MC_DROPOUT_RUNS):
            logits = mc_model.predict(data)
            accuracies.append(accuracy(logits, labels))
            all_logits[i, :, :] = logits

        # Compute average, variance, and uncertainty of logits
        logits = np.mean(all_logits, axis=0)
        save_evaluation_files(labels, logits, classes, "mc_dropout", FINAL_OUTPUT_DIR)

    if TEST_TIME_AUGMENTATION:
        NUM_TEST_TIME_AUGMENTATION_RUNS = 20
        print(f"Running {NUM_TEST_TIME_AUGMENTATION_RUNS} runs of Test Time Augmentation")
        print("========================")

        # Setup model with augmentation output
        model_path = os.path.join(MODEL_DIR, "model-0", MODEL_FILE)
        print(f"Looking for model at {model_path}")
        model = tf.keras.models.load_model(model_path)
        augmentation = ImageDataGenerator(
            rotation_range=10,
            fill_mode='nearest',
            horizontal_flip=True,
            vertical_flip=False,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        augmented_image_generator = augmentation.flow(data, labels, shuffle=False, batch_size=1)

        # Compute logits
        all_logits = np.zeros((NUM_TEST_TIME_AUGMENTATION_RUNS, labels.shape[0], labels.shape[1]))
        accuracies = []
        for i in range(NUM_TEST_TIME_AUGMENTATION_RUNS):
            logits = model.predict(augmented_image_generator, steps=data.shape[0])
            accuracies.append(accuracy(logits, labels))
            all_logits[i, :, :] = logits

        # Compute average, variance, and uncertainty of logits
        logits = np.mean(all_logits, axis=0)
        save_evaluation_files(labels, logits, classes, "test_time_augmentation", FINAL_OUTPUT_DIR)

    if DEEP_ENSEMBLE:
        # Find paths to models
        print("Starting Deep Ensemble")
        print("========================")
        print(f"Looking in {MODEL_DIR}")
        num_models = len(os.listdir(MODEL_DIR))
        model_filenames = [os.path.join(MODEL_DIR, f"model-{i}", MODEL_FILE) for i in range(num_models)]
        print(f"Found {num_models} items. Looking at {model_filenames}")

        # Compute logits
        all_logits = np.zeros((num_models, labels.shape[0], labels.shape[1]))
        accuracies = []
        for i, model_filename in enumerate(model_filenames):
            one_model = tf.keras.models.load_model(model_filename)
            logits = one_model.predict(data)
            accuracies.append(accuracy(logits, labels))
            all_logits[i, :, :] = logits

        # Compute average, variance, and uncertainty of logits
        average_logits = np.mean(all_logits, axis=0)
        save_evaluation_files(labels, logits, classes, "deep_ensemble", FINAL_OUTPUT_DIR)
