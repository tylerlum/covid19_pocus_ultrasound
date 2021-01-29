import wandb
import argparse
import os
import random
import imgaug
import pickle
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

from pocovidnet.video_augmentation import DataGenerator

from pocovidnet import VIDEO_MODEL_FACTORY
from pocovidnet.videoto3d import Videoto3D
from pocovidnet.wandb import WandbClassificationCallback, wandb_log_classification_table_and_plots
from datetime import datetime
from datetime import date


warnings.filterwarnings("ignore")
datestring = date.today().strftime("%b-%d-%Y") + "_" + datetime.now().strftime('%H-%M-%S')


def set_random_seed(seed_value):
    print(f"Setting random seed with {seed_value}")
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    imgaug.random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition'
    )
    # Input and output parameters
    parser.add_argument(
        '--videos',
        type=str,
        default='../data/pocus_videos/convex',
        help='directory where videos are stored'
    )
    parser.add_argument('--load', action='store_true')

    # Options for viewing
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save', action='store_true')

    # Wandb setup
    parser.add_argument('--wandb_project', type=str, default="covid-video-debugging")

    # Random seed
    parser.add_argument('--random_seed', type=int, default=1233)

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--frame_rate', type=int, default=5)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--optical_flow', action='store_true')
    parser.add_argument('--architecture', type=str, default="2D_CNN_average")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--optimizer', type=str, default="adam")

    parser.add_argument('--reduce_learning_rate', action='store_true')
    parser.add_argument('--reduce_learning_rate_monitor', type=str, default="val_loss")
    parser.add_argument('--reduce_learning_rate_mode', type=str, default="min")
    parser.add_argument('--reduce_learning_rate_factor', type=float, default=0.1)
    parser.add_argument('--reduce_learning_rate_patience', type=int, default=7)

    args = parser.parse_args()

    # Deterministic behavior
    set_random_seed(args.random_seed)

    # Output directory
    OUTPUT_DIR = "video_model_outputs"
    if not os.path.isdir(OUTPUT_DIR):
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
    FINAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, datestring)
    if not os.path.exists(FINAL_OUTPUT_DIR):
        os.makedirs(FINAL_OUTPUT_DIR)

    SAVE_DIR = '../data/video_input_data/'
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    train_save_path, validation_save_path, test_save_path = (os.path.join(SAVE_DIR, "conv3d_train.dat"),
                                                             os.path.join(SAVE_DIR, "conv3d_validation.dat"),
                                                             os.path.join(SAVE_DIR, "conv3d_test.dat"))

    # Load saved data or read in videos
    print()
    print("===========================")
    print("Converting videos to 4D video clips")
    print("===========================")
    if args.load:
        with open(test_save_path, 'rb') as infile:
            X_test, test_labels_text, test_files = pickle.load(infile)
        with open(validation_save_path, 'rb') as infile:
            X_validation, validation_labels_text, validation_files = pickle.load(infile)
        with open(train_save_path, 'rb') as infile:
            X_train, train_labels_text, train_files = pickle.load(infile)
    else:
        # SPLIT NO CROSSVAL
        class_short = ["cov", "pne", "reg"]
        vid_files = [
            v for v in os.listdir(args.videos) if v[:3].lower() in class_short
        ]
        labels = [vid[:3].lower() for vid in vid_files]
        train_files, test_files, train_labels, test_labels = train_test_split(
            vid_files, labels, stratify=labels, test_size=0.2, random_state=args.random_seed
        )
        train_files, validation_files, train_labels, validation_labels = train_test_split(
            train_files, train_labels, stratify=train_labels, test_size=0.2, random_state=args.random_seed
        )

        # Read in videos and transform to 3D
        vid3d = Videoto3D(args.videos, width=args.width, height=args.height, depth=args.depth, framerate=args.frame_rate, grayscale=args.grayscale, optical_flow=args.optical_flow)
        if not args.save:
            train_save_path, validation_save_path, test_save_path = None, None, None
        X_train, train_labels_text, train_files = vid3d.video3d(
            train_files,
            train_labels,
            save=train_save_path
        )
        X_validation, validation_labels_text, validation_files = vid3d.video3d(
            validation_files,
            validation_labels,
            save=validation_save_path
        )
        X_test, test_labels_text, test_files = vid3d.video3d(
            test_files,
            test_labels,
            save=test_save_path
        )

    # One-hot encoding
    lb = LabelBinarizer()
    lb.fit(train_labels_text)
    Y_train = lb.transform(train_labels_text)
    Y_validation = np.array(lb.transform(validation_labels_text))
    Y_test = np.array(lb.transform(test_labels_text))

    # Model genesis requires different dataset shape than VGG. Requires width = height = 64, grayscale
    if args.architecture == "model_genesis":
        # Rearrange to put channels first and depth last
        X_train = np.transpose(X_train, [0, 4, 2, 3, 1])
        X_validation = np.transpose(X_validation, [0, 4, 2, 3, 1])
        X_test = np.transpose(X_test, [0, 4, 2, 3, 1])

        # Repeat frames since depth of model is 32
        X_train = np.repeat(X_train, [6, 7, 7, 6, 6], axis=-1)
        X_validation = np.repeat(X_validation, [6, 7, 7, 6, 6], axis=-1)
        X_test = np.repeat(X_test, [6, 7, 7, 6, 6], axis=-1)

    input_shape = X_train.shape[1:]
    print(f"input_shape = {input_shape}")

    generator = DataGenerator(X_train, Y_train, args.batch_size, input_shape, lb.classes_, shuffle=False)

    # VISUALIZE
    if args.visualize:
        for i in range(X_train.shape[0]):
            example = X_train[i]
            label = Y_train[i]
            print(f"Label = {label}")
            for j in range(example.shape[0]):
                print(f"Frame {j}")
                frame = example[j]
                print(f"np.max(frame) = {np.max(frame)}")
                print(f"np.min(frame) = {np.min(frame)}")
                cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Example-{i}_Frame-{j}_Label-{label}.jpg"), 255*frame)
            if i > 8:
                break

        batchX, batchY = generator[0]
        i = 0
        for frames, label in zip(batchX, batchY):
            j = 0
            for frame in frames:
                cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Augmented-Example-{i}_Frame-{j}_Label-{label}.jpg"), 255*frame)
                j += 1
            i += 1

    # Verbose
    print()
    print("===========================")
    print("Printing details about dataset")
    print("===========================")
    print(X_train.shape, Y_train.shape)
    print(X_validation.shape, Y_validation.shape)
    print(X_test.shape, Y_test.shape)
    nb_classes = len(np.unique(train_labels_text))
    print(nb_classes, np.max(X_train))
    train_uniques, train_counts = np.unique(train_labels_text, return_counts=True)
    validation_uniques, validation_counts = np.unique(validation_labels_text, return_counts=True)
    test_uniques, test_counts = np.unique(test_labels_text, return_counts=True)
    print("unique in train", (train_uniques, train_counts))
    print("unique in validation", (validation_uniques, validation_counts))
    print("unique in test", (test_uniques, test_counts))

    class_weight = {i: sum(train_counts) / train_counts[i] for i in range(len(train_counts))}
    print(f"class_weight = {class_weight}")

    model = VIDEO_MODEL_FACTORY[args.architecture](input_shape, nb_classes)

    tf.keras.utils.plot_model(model, os.path.join(FINAL_OUTPUT_DIR, f"{args.architecture}.png"), show_shapes=True)

    if args.optimizer == "adam":
        opt = Adam(lr=args.learning_rate)
    else:
        print(f"WARNING: invalid optimizer {args.optimizer}")

    model.compile(
        optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy']
    )

    # Define callbacks
    reduce_learning_rate_loss = ReduceLROnPlateau(
        monitor=args.reduce_learning_rate_monitor,
        factor=args.reduce_learning_rate_factor,
        patience=args.reduce_learning_rate_patience,
        mode=args.reduce_learning_rate_mode,
        verbose=1,
        epsilon=1e-4,
    )

    wandb.init(entity='tylerlum', project=args.wandb_project)
    wandb.config.update(args)
    wandb.config.final_output_dir = FINAL_OUTPUT_DIR

    wandb_callback = WandbClassificationCallback(log_confusion_matrix=True, confusion_classes=len(lb.classes_), validation_data=(X_validation, Y_validation), labels=lb.classes_)

    callbacks = [wandb_callback]
    if args.reduce_learning_rate:
        callbacks.append(reduce_learning_rate_loss)

    print()
    print("===========================")
    print("About to train model")
    print("===========================")
    print(model.summary())

    if args.augment:
        H = model.fit(
            generator,
            validation_data=(X_validation, Y_validation),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
            shuffle=False,
            class_weight=class_weight,
            callbacks=callbacks,
        )
    else:
        H = model.fit(
            X_train, Y_train,
            validation_data=(X_validation, Y_validation),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
            shuffle=False,
            class_weight=class_weight,
            callbacks=callbacks,
        )

    print()
    print("===========================")
    print("Evaluating network...")
    print("===========================")
    # Can cause out of memory issue when using larger framerate
    trainLoss, trainAcc = model.evaluate(X_train, Y_train, verbose=1)
    print('train loss:', trainLoss)
    print('train accuracy:', trainAcc)
    validationLoss, validationAcc = model.evaluate(X_validation, Y_validation, verbose=1)
    print('Validation loss:', validationLoss)
    print('Validation accuracy:', validationAcc)
    testLoss, testAcc = model.evaluate(X_test, Y_test, verbose=1)
    print('Test loss:', testLoss)
    print('Test accuracy:', testAcc)

    trainPredIdxs = model.predict(X_train, batch_size=args.batch_size)
    validationPredIdxs = model.predict(X_validation, batch_size=args.batch_size)
    testPredIdxs = model.predict(X_test, batch_size=args.batch_size)

    def savePredictionsToCSV(predIdxs, csvFilename, directory=FINAL_OUTPUT_DIR):
        df = pd.DataFrame(predIdxs)
        df.to_csv(os.path.join(directory, csvFilename))
    savePredictionsToCSV(validationPredIdxs, "validation_preds_last_epoch.csv")
    savePredictionsToCSV(testPredIdxs, "test_preds_last_epoch.csv")

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    trainPredIdxs = np.argmax(trainPredIdxs, axis=1)
    validationPredIdxs = np.argmax(validationPredIdxs, axis=1)
    testPredIdxs = np.argmax(testPredIdxs, axis=1)

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    def printAndSaveClassificationReport(y, predIdxs, classes, reportFilename, directory=FINAL_OUTPUT_DIR):
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

        wandb_log_classification_table_and_plots(report, reportFilename)

    printAndSaveClassificationReport(Y_train, trainPredIdxs, lb.classes_, "trainReport.csv")
    printAndSaveClassificationReport(Y_validation, validationPredIdxs, lb.classes_, "validationReport.csv")
    printAndSaveClassificationReport(Y_test, testPredIdxs, lb.classes_, "testReport.csv")

    def printAndSaveConfusionMatrix(y, predIdxs, classes, confusionMatrixFilename, directory=FINAL_OUTPUT_DIR):
        print(f'confusion matrix for {confusionMatrixFilename}')

        cm = confusion_matrix(y.argmax(axis=1), predIdxs)
        # show the confusion matrix, accuracy, sensitivity, and specificity
        print(cm)

        plt.figure()
        cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        cmDisplay.plot()
        plt.savefig(os.path.join(directory, confusionMatrixFilename))
    printAndSaveConfusionMatrix(Y_train, trainPredIdxs, lb.classes_, "trainConfusionMatrix.png")
    printAndSaveConfusionMatrix(Y_validation, validationPredIdxs, lb.classes_, "validationConfusionMatrix.png")
    printAndSaveConfusionMatrix(Y_test, testPredIdxs, lb.classes_, "testConfusionMatrix.png")

    # print(f'Saving COVID-19 detector model on {FINAL_OUTPUT_DIR} data...')
    # model.save(os.path.join(FINAL_OUTPUT_DIR, 'last_epoch'), save_format='h5')

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
    plt.savefig(os.path.join(FINAL_OUTPUT_DIR, 'loss.png'))

    def calculate_patient_wise(files, x, y, model):
        gt = []
        preds = []
        files = np.array(files)
        for video in np.unique(files):
            current_data = x[files == video]
            current_labels = y[files == video]
            true_label = current_labels[0]
            current_predictions = model.predict(current_data)
            prediction = np.argmax(np.mean(current_predictions, axis=0))
            gt.append(true_label)
            preds.append(prediction)
            print(f"video = {video}, true_label = {true_label}, prediction = {prediction}")
        return np.array(gt), np.array(preds)
    print("-----------------------------TRAINING-----------------------------")
    train_gt, train_preds = calculate_patient_wise(train_files, X_train, Y_train, model)
    print("-----------------------------VALIDATION-----------------------------")
    validation_gt, validation_preds = calculate_patient_wise(validation_files, X_validation, Y_validation, model)
    print("-----------------------------TESTING-----------------------------")
    test_gt, test_preds = calculate_patient_wise(test_files, X_test, Y_test, model)

    printAndSaveClassificationReport(train_gt, train_preds, lb.classes_, "trainReportPatients.csv")
    printAndSaveClassificationReport(validation_gt, validation_preds, lb.classes_, "validationReportPatients.csv")
    printAndSaveClassificationReport(test_gt, test_preds, lb.classes_, "testReportPatients.csv")
    printAndSaveConfusionMatrix(train_gt, train_preds, lb.classes_, "trainConfusionMatrixPatients.png")
    printAndSaveConfusionMatrix(validation_gt, validation_preds, lb.classes_, "validationConfusionMatrixPatients.png")
    printAndSaveConfusionMatrix(test_gt, test_preds, lb.classes_, "testConfusionMatrixPatients.png")

    # del X_train, X_validation, Y_train, Y_validation
    def run_on_all_frames():
        for test_file, test_label in zip(test_files, test_labels):
            cap = cv2.VideoCapture(os.path.join(args.videos, test_file))

            current_data = []
            video_num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            while cap.isOpened():
                frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
                ret, frame = cap.read()
                if (ret != True):
                    break

                image = frame if not args.grayscale else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (args.width, args.height))

                current_data.append(image)

                # Store video clip
                reached_last_frame = (frame_id == video_num_frames - 1)
                if reached_last_frame:
                    model_input = np.asarray(current_data) / 255
                    print(model_input.shape)
                    print(test_label)
                    sdfd = model.predict(np.array([model_input]))
                    print(sdfd)
                    print("----------")
                    current_data = []

            cap.release()

if __name__ == '__main__':
    main()
