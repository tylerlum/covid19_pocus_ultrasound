import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import math
import sys
from tqdm import tqdm
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
from datetime import datetime
from datetime import date


def save_evaluation_files(labels, logits, classes, start_of_filename, directory):
    print("1")
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

    print("2")
    # make predictions on the testing set
    savePredictionsToCSV(logits, start_of_filename, directory)
    print("3")

    predIdxs = np.argmax(logits, axis=1)
    print(labels.shape)
    print(predIdxs.shape)
    print(len(classes))
    print("4")
    printAndSaveClassificationReport(labels, predIdxs, classes, start_of_filename, directory)
    print("5")
    printAndSaveConfusionMatrix(labels, predIdxs, classes, start_of_filename, directory)



def accuracy(logits, labels):
    correct = np.sum(np.argmax(labels, axis=1) == np.argmax(logits, axis=1))
    total = labels.shape[0]
    return correct / total


def create_mc_model(model, dropProb=0.5):
  layers = [l for l in model.layers]
  x = layers[0].output
  for i in range(1, len(layers)):
      # Replace dropout layers with MC dropout layers
      if isinstance(layers[i], Dropout):
          x = Dropout(dropProb)(x, training=True)
      else:
          x = layers[i](x)
  mc_model = Model(inputs=layers[0].input, outputs=x)
  mc_model.set_weights(model.get_weights())
  return mc_model


def get_patientwise_dataset():
    # Get patient images and labels from the videos
    patient_image_lists, patient_label_lists = [], []
    image_counter = 0
    FULL_VIDEOS_DIR = DATA_DIR + f"_full_videos/split{FOLD}"
    class_dirs = [x[0] for x in os.walk(FULL_VIDEOS_DIR)]
    for class_dir in class_dirs:
        for _, _, files in os.walk(class_dir):
            for file_ in files:
                video_path = os.path.join(class_dir,  file_)
                if os.path.exists(video_path):
                    path_parts = video_path.split(os.path.sep)
                    label = path_parts[-2]
                    cap = cv2.VideoCapture(video_path)
                    patient_image_list, patient_label_list = [], []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if (ret != True):
                            break
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                        patient_image_list.append(image)
                        patient_label_list.append(label)
                        image_counter += 1
                    patient_label_lists.append(np.array(patient_label_list))
                    patient_image_lists.append(np.array(patient_image_list) / 255)
    print(f"Got {image_counter} images from {len(patient_image_lists)} videos")
    possible_labels = list(set([label_list[0] for label_list in patient_label_lists]))
    num_classes = len(possible_labels)

    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    lb.fit(possible_labels)

    patient_label_lists = [lb.transform(patient_label_list) for patient_label_list in patient_label_lists]

    if num_classes == 2:
        patient_label_lists = [to_categorical(patient_label_list, num_classes=num_classes) for patient_label_list in patient_label_lists]
    return patient_image_lists, patient_label_lists, lb.classes_



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

    print(f'Evaluating with: {args}')

    # Setup output dir
    datestring = date.today().strftime("%b-%d-%Y") + "_" + datetime.now().strftime('%H-%M-%S')
    FINAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f"{__file__}__Data-{DATA_DIR}__Model-{MODEL_DIR}__Fold-{FOLD}__MC-{MC_DROPOUT}__TTA-{TEST_TIME_AUGMENTATION}__ENSEMBLE-{DEEP_ENSEMBLE}__".replace('/', '|') + datestring)
    if not os.path.exists(FINAL_OUTPUT_DIR):
        os.makedirs(FINAL_OUTPUT_DIR)

    # Get dataset
    patient_image_lists, patient_label_lists, classes = get_patientwise_dataset()
    patient_image_lists, patient_label_lists = patient_image_lists[0:4] + patient_image_lists[15:19] + patient_image_lists[26:], patient_label_lists[0:4] + patient_label_lists[15:19] + patient_label_lists[26:]

    # Setup model
    if MC_DROPOUT:
        NUM_MC_DROPOUT_RUNS = 50
        print(f"Running {NUM_MC_DROPOUT_RUNS} runs of MC Dropout")
        print("========================")

        # Create MC model
        model_path = os.path.join(MODEL_DIR, "model-0", MODEL_FILE)
        print(f"Looking for model at {model_path}")
        model = tf.keras.models.load_model(model_path)
        mc_model = create_mc_model(model)

        # Get patient-wise data
        accuracies = []
        patientwise_labels = []
        patientwise_average_logits = []
        patientwise_weighted_average_logits = []
        for j, (imgs_, labels_) in enumerate(zip(patient_image_lists, patient_label_lists)):
            all_logits_single_patient = np.zeros((NUM_MC_DROPOUT_RUNS, labels_.shape[0], labels_.shape[1]))
            for i in tqdm(range(NUM_MC_DROPOUT_RUNS)):
                logits = mc_model.predict(imgs_)
                accuracies.append(accuracy(logits, labels_))
                all_logits_single_patient[i, :, :] = logits

            average_logits_single_patient = np.mean(all_logits_single_patient, axis=0)
            std_dev_logits_single_patient = np.std(all_logits_single_patient, axis=0, ddof=1)
            indices_of_prediction_single_patient = np.argmax(average_logits_single_patient, axis=1)
            uncertainty_in_prediction = np.take_along_axis(std_dev_logits_single_patient, np.expand_dims(indices_of_prediction_single_patient, axis=1), axis=-1).squeeze(axis=-1)

            # Method 1: Use only the highest predicted value for uncertainty weighting
            # weighted_logits = [0] * 3
            # for i in range(uncertainty_in_prediction.shape[0]):
            #     # weighted_logits[indices_of_prediction_single_patient[i]] += 1/uncertainty_in_prediction[i]
            #     weighted_logits[indices_of_prediction_single_patient[i]] += math.exp(-uncertainty_in_prediction[i])
            # average_weighted_logits = np.array(weighted_logits) / sum(weighted_logits)

            # Method 2: Use all predicted values for uncertainty weighting
            weighted_logits = np.divide(average_logits_single_patient, std_dev_logits_single_patient + 0.000001)
            # weighted_logits = np.multiply(average_logits_single_patient, np.exp(-std_dev_logits_single_patient))
            average_weighted_logits = np.mean(weighted_logits, axis=0)
            average_weighted_logits = average_weighted_logits / np.sum(average_weighted_logits)
            average_logits = np.mean(average_logits_single_patient, axis=0)

            # Print logits at each frame
            # for i in range(average_logits_single_patient.shape[0]):
            #     print(f"average_logits_single_patient = {average_logits_single_patient[i]}")
            #     print(f"std_dev_logits_single_patient = {std_dev_logits_single_patient[i]}")

            # Print patient-wise prediction
            print(f"----------------------")
            print(j)
            print(f"average_logits = {average_logits}")
            print(f"average_weighted_logits = {average_weighted_logits}")
            print(f"ground_truth = {labels_[0]}")
            print(f"======================")
            patientwise_labels.append(labels_[0])
            patientwise_average_logits.append(average_logits)
            patientwise_weighted_average_logits.append(average_weighted_logits)

            # Save first frames to visualize
            # for i in range(imgs_.shape[0]):
            #     cv2.imwrite(f"img_{j}_{i}.jpg", imgs_[i]*255)
            #     break
        print(classes)
        save_evaluation_files(np.array(patientwise_labels), np.array(patientwise_average_logits), classes, "patientwiseAverage", FINAL_OUTPUT_DIR)
        save_evaluation_files(np.array(patientwise_labels), np.array(patientwise_weighted_average_logits), classes, "patientwiseAverageWeighted", FINAL_OUTPUT_DIR)


    if TEST_TIME_AUGMENTATION:
        NUM_TEST_TIME_AUGMENTATION_RUNS = 50
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
        for i in tqdm(range(NUM_TEST_TIME_AUGMENTATION_RUNS)):
            logits = model.predict(augmented_image_generator, steps=data.shape[0])
            accuracies.append(accuracy(logits, labels))
            all_logits[i, :, :] = logits

        # Compute average, variance, and uncertainty of logits
        average_logits = np.mean(all_logits, axis=0)
        std_dev_logits = np.std(all_logits, axis=0, ddof=1)
        indices_of_prediction = np.argmax(average_logits, axis=1)
        uncertainty_in_prediction = np.take_along_axis(std_dev_logits, np.expand_dims(indices_of_prediction, axis=1), axis=-1).squeeze(axis=-1)
        print(f"Mean accuracy of individual tta models = {sum(accuracies)/len(accuracies)}")
        print(f"Combined accuracy of tta models = {accuracy(average_logits, labels)}")
        print(f"Average uncertainty in tta predictions = {np.sum(uncertainty_in_prediction)/uncertainty_in_prediction.shape[0]}")

        # Plot accuracy vs uncertainty
        correct_labels = np.take_along_axis(labels, np.expand_dims(indices_of_prediction, axis=1), axis=-1).squeeze(axis=-1)
        l1_loss_of_prediction = np.absolute(correct_labels - np.max(average_logits, axis=1))
        plot_loss_vs_uncertainty(l1_loss_of_prediction, uncertainty_in_prediction, start_of_filename="test_time_augmentation")

        # Plot RAR vs RER
        prediction_accuracies = np.argmax(labels, axis=1) == np.argmax(average_logits, axis=1)
        plot_rar_vs_rer(prediction_accuracies, uncertainty_in_prediction, start_of_filename="test_time_augmentation")

        # Save relevant images
        save_special_images(data, labels, l1_loss_of_prediction, uncertainty_in_prediction, num_images=20, start_of_filename="test_time_augmentation")

        # Save max/min loss/certainty images
        save_max_min_loss_certainty_images(data, labels, l1_loss_of_prediction, uncertainty_in_prediction, num_images=20, start_of_filename="test_time_augmentation")
        plot_histogram_labels(data, labels, l1_loss_of_prediction, uncertainty_in_prediction, start_of_filename="test_time_augmentation")

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
        for i in tqdm(range(len(model_filenames))):
            model_filename = model_filenames[i]
            one_model = tf.keras.models.load_model(model_filename)
            logits = one_model.predict(data)
            accuracies.append(accuracy(logits, labels))
            all_logits[i, :, :] = logits

        # Compute average, variance, and uncertainty of logits
        average_logits = np.mean(all_logits, axis=0)
        std_dev_logits = np.std(all_logits, axis=0, ddof=1)
        indices_of_prediction = np.argmax(average_logits, axis=1)
        uncertainty_in_prediction = np.take_along_axis(std_dev_logits, np.expand_dims(indices_of_prediction, axis=1), axis=-1).squeeze(axis=-1)

        print(f"Mean accuracy of individual deep ensemble models = {sum(accuracies)/len(accuracies)}")
        print(f"Combined accuracy of deep ensemble models = {accuracy(average_logits, labels)}")
        print(f"Average uncertainty in deep ensemble prediction = {np.sum(uncertainty_in_prediction)/uncertainty_in_prediction.shape[0]}")

        # Plot accuracy vs uncertainty
        correct_labels = np.take_along_axis(labels, np.expand_dims(indices_of_prediction, axis=1), axis=-1).squeeze(axis=-1)
        l1_loss_of_prediction = np.absolute(correct_labels - np.max(average_logits, axis=1))
        plot_loss_vs_uncertainty(l1_loss_of_prediction, uncertainty_in_prediction, start_of_filename="deep_ensemble")

        # Plot RAR vs RER
        prediction_accuracies = np.argmax(labels, axis=1) == np.argmax(average_logits, axis=1)
        plot_rar_vs_rer(prediction_accuracies, uncertainty_in_prediction, start_of_filename="deep_ensemble")

        # Save relevant images
        save_special_images(data, labels, l1_loss_of_prediction, uncertainty_in_prediction, num_images=20, start_of_filename="deep_ensemble")

        # Save max/min loss/certainty images
        save_max_min_loss_certainty_images(data, labels, l1_loss_of_prediction, uncertainty_in_prediction, num_images=20, start_of_filename="deep_ensemble")

        plot_histogram_labels(data, labels, l1_loss_of_prediction, uncertainty_in_prediction, start_of_filename="deep_ensemble")
