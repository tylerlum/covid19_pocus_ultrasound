seed_value = 1231
import argparse
import math
import json
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import sys
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
import pickle
import warnings

import numpy as np
np.random.seed(seed_value)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.random.set_seed(seed_value)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling3D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence

from pocovidnet.utils import fix_layers

from pocovidnet import VIDEO_MODEL_FACTORY
from pocovidnet.videoto3d import Videoto3D
from datetime import datetime
from datetime import date
from vidaug import augmentors as va


warnings.filterwarnings("ignore")
datestring = date.today().strftime("%b-%d-%Y") + "_" + datetime.now().strftime('%H-%M-%S')


def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition'
    )
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument(
        '--videos',
        type=str,
        default='../data/pocus_videos/convex',
        help='directory where videos are stored'
    )
    parser.add_argument(
        '--json', type=str, default="../data/cross_val.json"
    )
    parser.add_argument('--output', type=str, default="video_model_outputs")
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--fr', type=int, default=5)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--model_id', type=str, default='base')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--trainable_base_layers', type=int, default=0)
    parser.add_argument(
        '--save', type=str, default='../data/video_input_data/'
    )
    parser.add_argument(
        '--weight_path', type=str, default='../Genesis_Chest_CT.h5'
    )

    args = parser.parse_args()

    # Out model directory
    MODEL_D = args.output
    if not os.path.isdir(MODEL_D):
        if not os.path.exists(MODEL_D):
            os.makedirs(args.output)
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    FINAL_OUTPUT_DIR = os.path.join(MODEL_D, datestring)
    if not os.path.exists(FINAL_OUTPUT_DIR):
        os.makedirs(FINAL_OUTPUT_DIR)

    # Load saved data or read in videos
    if args.load:
        with open(
            os.path.join(
                args.save, 'conv3d_test_fold_' + str(args.fold) + '.dat'
            ), 'rb'
        ) as infile:
            X_test, test_labels_text, test_files = pickle.load(infile)
        with open(
            os.path.join(
                args.save, 'conv3d_validation_fold_' + str(args.fold) + '.dat'
            ), 'rb'
        ) as infile:
            X_validation, validation_labels_text, validation_files = pickle.load(infile)
        with open(
            os.path.join(
                args.save, 'conv3d_train_fold_' + str(args.fold) + '.dat'
            ), 'rb'
        ) as infile:
            X_train, train_labels_text, train_files = pickle.load(infile)
    else:
        # SPLIT NO CROSSVAL
        class_short = ["cov", "pne", "reg"]
        vid_files = [
            v for v in os.listdir(args.videos) if v[:3].lower() in class_short
        ]
        labels = [vid[:3].lower() for vid in vid_files]
        train_files, test_files, train_labels, test_labels = train_test_split(
            vid_files, labels, stratify=labels, test_size=0.2
        )
        train_files, validation_files, train_labels, validation_labels = train_test_split(
            train_files, train_labels, stratify=train_labels, test_size=0.2
        )
        full_video_train_files, full_video_validation_files, full_video_test_files, full_video_train_labels, full_video_validation_labels, full_video_test_labels = train_files, validation_files, test_files, train_labels, validation_labels, test_labels

        # Read in videos and transform to 3D
        vid3d = Videoto3D(args.videos, width=224, height=224, depth=args.depth, framerate=args.fr)
        X_train, train_labels_text, train_files = vid3d.video3d(
            train_files,
            train_labels,
            save=os.path.join(
                args.save, "conv3d_train_fold_" + str(args.fold) + ".dat"
            )
        )
        X_validation, validation_labels_text, validation_files = vid3d.video3d(
            validation_files,
            validation_labels,
            save=os.path.join(
                args.save, "conv3d_validation_fold_" + str(args.fold) + ".dat"
            )
        )
        X_test, test_labels_text, test_files = vid3d.video3d(
            test_files,
            test_labels,
            save=os.path.join(
                args.save, "conv3d_test_fold_" + str(args.fold) + ".dat"
            )
        )

    # One-hot encoding
    lb = LabelBinarizer()
    lb.fit(train_labels_text)
    Y_train = lb.transform(train_labels_text)
    Y_validation = np.array(lb.transform(validation_labels_text))
    Y_test = np.array(lb.transform(test_labels_text))

    ## VISUALIZE
    if args.visualize:
        for i in range(20):
            # 'video' should be either a list of images from type of numpy array or PIL images
            orig_video = X_train[i]
            video = np.squeeze(orig_video)
            video = [video, video, video]
            video = np.stack(video, axis=3)
            print(video.shape)
            video_aug = seq(video*255)
            print(len(video_aug))
            print(video_aug[0].shape)
            for j in range(1):
                import cv2
                print(f"Frame {j}")
                frame = video_aug[j]
                print(f"np.max(frame) = {np.max(frame)}")
                print(f"np.min(frame) = {np.min(frame)}")
                cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"video_aug-{i}_Frame-{j}.jpg"), frame)
                cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"video_aug-{i}_Frame-{j}org.jpg"), 255*orig_video[j])

        for i in range(X_train.shape[0]):
            example = X_train[i]
            label = Y_train[i]
            print(f"Label = {label}")
            for j in range(example.shape[0]):
                import cv2
                print(f"Frame {j}")
                frame = example[j]
                print(f"np.max(frame) = {np.max(frame)}")
                print(f"np.min(frame) = {np.min(frame)}")
                cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Example-{i}_Frame-{j}_Label-{label}.jpg"), 255*frame)


    # Verbose
    print("testing on split", args.fold)
    print(X_train.shape, Y_train.shape)
    print(X_validation.shape, Y_validation.shape)
    print(X_test.shape, Y_test.shape)
    nb_classes = len(np.unique(train_labels_text))
    print(nb_classes, np.max(X_train))
    print("unique in train", np.unique(train_labels_text, return_counts=True))
    print("unique in validation", np.unique(validation_labels_text, return_counts=True))
    print("unique in test", np.unique(test_labels_text, return_counts=True))

    # Define callbacks
    if args.model_id == 'base':
        input_shape = X_train.shape[1:]
        print(f"input_shape = {input_shape}")
        model = VIDEO_MODEL_FACTORY[args.model_id](input_shape, nb_classes)
        tf.keras.utils.plot_model(model, "my_first_model.png", show_shapes=True)

    opt = Adam(lr=args.lr)
    model.compile(
        optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy']
    )
    class_weight = {0: 1.,
                    1: 2.,
                    2: 1.}

    # Define callbacks
    earlyStopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=7,
        verbose=1,
        epsilon=1e-4,
        mode='min'
    )
    print(model.summary())

    H = model.fit(
        X_train, Y_train,
        validation_data=(X_validation, Y_validation),
        epochs=args.epoch,
        batch_size=args.batch,
        verbose=1,
        shuffle=False,
        class_weight=class_weight,
        use_multiprocessing=True,
        workers=2,  # Empirically best performance
        callbacks=[earlyStopping, reduce_lr_loss],
    )

    print('Evaluating network...')
    trainLoss, trainAcc = model.evaluate(X_train, Y_train, verbose=0)
    print('train loss:', trainLoss)
    print('train accuracy:', trainAcc)
    validationLoss, validationAcc = model.evaluate(X_validation, Y_validation, verbose=0)
    print('Validation loss:', validationLoss)
    print('Validation accuracy:', validationAcc)
    testLoss, testAcc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', testLoss)
    print('Test accuracy:', testAcc)

    trainPredIdxs = model.predict(X_train, batch_size=args.batch)
    validationPredIdxs = model.predict(X_validation, batch_size=args.batch)
    testPredIdxs = model.predict(X_test, batch_size=args.batch)
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

    print(f'Saving COVID-19 detector model on {FINAL_OUTPUT_DIR} data...')
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


    # going patient-wise #################
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
        return np.array(gt), np.array(preds)
    train_gt, train_preds = calculate_patient_wise(train_files, X_train, Y_train, model)
    validation_gt, validation_preds = calculate_patient_wise(validation_files, X_validation, Y_validation, model)
    test_gt, test_preds = calculate_patient_wise(test_files, X_test, Y_test, model)

    printAndSaveClassificationReport(train_gt, train_preds, lb.classes_, "trainReportPatients.csv")
    printAndSaveClassificationReport(validation_gt, validation_preds, lb.classes_, "validationReportPatients.csv")
    printAndSaveClassificationReport(test_gt, test_preds, lb.classes_, "testReportPatients.csv")
    printAndSaveConfusionMatrix(train_gt, train_preds, lb.classes_, "trainConfusionMatrixPatients.png")
    printAndSaveConfusionMatrix(validation_gt, validation_preds, lb.classes_, "validationConfusionMatrixPatients.png")
    printAndSaveConfusionMatrix(test_gt, test_preds, lb.classes_, "testConfusionMatrixPatients.png")

if __name__ == '__main__':
    main()
