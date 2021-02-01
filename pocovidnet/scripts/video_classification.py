import wandb
import argparse
import os
import random
import imgaug
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

from pocovidnet.video_augmentation import DataGenerator

from pocovidnet import VIDEO_MODEL_FACTORY
from pocovidnet.videoto3d import Videoto3D
from pocovidnet.wandb import ConfusionMatrixEachEpochCallback, wandb_log_classification_table_and_plots
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition'
    )
    parser.add_argument('--wandb_project', type=str, default="covid-video-debugging", help='wandb project name')

    # Input files
    parser.add_argument(
        '--videos',
        type=str,
        default='../data/pocus_videos_Jan_30_2021/convex',
        help='directory where videos are stored'
    )

    # Output files
    parser.add_argument('--save_model', type=str2bool, nargs='?', const=True, default=False, help='save final model')
    parser.add_argument('--visualize', type=str2bool, nargs='?', const=True, default=False,
                        help='Save images to visualize in output dir')

    # Remove randomness
    parser.add_argument('--random_seed', type=int, default=1233, help='random seed for all randomness of the script')

    # K fold cross validation
    parser.add_argument('--num_folds', type=int, default=5, help='number of cross validation folds, splits up by file')
    parser.add_argument('--test_fold', type=int, default=0, help='fold for test. validation = (test_fold+1)%num_folds')

    # Save confusion matrix for each epoch
    parser.add_argument('--confusion_matrix_each_epoch', type=str2bool, nargs='?', const=True, default=False,
                        help='Save a confusion matrix to wandb at the end of each epoch')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    parser.add_argument('--frame_rate', type=int, default=5, help='framerate to get frames from videos into clips')
    parser.add_argument('--depth', type=int, default=5, help="number of frames per video clip")
    parser.add_argument('--width', type=int, default=224, help='video clip width')
    parser.add_argument('--height', type=int, default=224, help='video clip height')
    parser.add_argument('--grayscale', type=str2bool, nargs='?', const=True, default=False, help='gray video clips')
    parser.add_argument('--optical_flow_type', type=str, default="farneback",
                        help=('algorithm for optical flow (found in OPTICAL_FLOW_ALGORITHM_FACTORY). ' +
                              'only used for networks starting with 2stream, else is automatically set to None'))
    parser.add_argument('--architecture', type=str, default="2D_CNN_average",
                        help='neural network architecture (found in VIDEO_MODEL_FACTORY)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--augment', type=str2bool, nargs='?', const=True, default=False, help='video augmentation')
    parser.add_argument('--optimizer', type=str, default="adam", help='optimizer for training')
    parser.add_argument('--pretrained_cnn', type=str, default="vgg16", help='pretrained cnn architecture')
    parser.add_argument('--extra_dense', type=int, nargs='+', help='extra dense layers list of units')
    parser.add_argument('--use_pooling', type=str2bool, nargs='?', const=True, default=False, help='LSTM pooling')

    parser.add_argument('--reduce_learning_rate', type=str2bool, nargs='?', const=True, default=False,
                        help='use reduce learning rate callback')
    parser.add_argument('--reduce_learning_rate_monitor', type=str, default="val_loss",
                        help='reduce learning rate depending on this, only used if reduce_learning_rate is true')
    parser.add_argument('--reduce_learning_rate_mode', type=str, default="min",
                        help='reduce learning rate when monitor is min/max, only used if reduce_learning_rate is true')
    parser.add_argument('--reduce_learning_rate_factor', type=float, default=0.1,
                        help='reduce learning rate by this factor, only used if reduce_learning_rate is true')
    parser.add_argument('--reduce_learning_rate_patience', type=int, default=7,
                        help='reduce learning rate if happens for x epochs, only used if reduce_learning_rate is true')

    args = parser.parse_args()
    print(f"raw args = {args}")

    print()
    print("===========================")
    print("Cleaning arguments")
    print("===========================")

    # Turn on optical flow only if needed
    if not args.architecture.startswith("2stream"):
        print("Not using optical flow")
        args.optical_flow_type = None

    # This model requires width = height = 64, grayscale
    if args.architecture == "model_genesis":
        args.grayscale = True
        args.width, args.height = 64, 64
        print("This model requires width, height, grayscale = {args.width}, {args.height}, {args.grayscale}")

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

    # Get videos and labels
    class_short = ["cov", "pne", "reg"]
    vid_files = [
        v for v in os.listdir(args.videos) if v[:3].lower() in class_short
    ]
    labels = [vid[:3].lower() for vid in vid_files]

    # Setup folds
    args.validation_fold = (args.test_fold + 1) % args.num_folds  # Select validation fold
    print()
    print("===========================")
    print(f"Performing k-fold splitting with validation fold {args.validation_fold} and test fold {args.test_fold}")
    print("===========================")
    k_fold_cross_validation = StratifiedKFold(n_splits=args.num_folds, random_state=args.random_seed, shuffle=True)

    def get_train_validation_test_split(validation_fold, test_fold, k_fold_cross_validation, vid_files, labels):
        for i, (train_index, test_index) in enumerate(k_fold_cross_validation.split(vid_files, labels)):
            if i == args.validation_fold:
                validation_indices = test_index
            elif i == args.test_fold:
                test_indices = test_index
        train_indices = [i for i in range(len(vid_files))
                         if i not in validation_indices and i not in test_indices]  # Need to use only remaining

        train_files = [vid_files[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        validation_files = [vid_files[i] for i in validation_indices]
        validation_labels = [labels[i] for i in validation_indices]
        test_files = [vid_files[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        return train_files, train_labels, validation_files, validation_labels, test_files, test_labels

    train_files, train_labels, validation_files, validation_labels, test_files, test_labels = (
            get_train_validation_test_split(args.validation_fold, args.test_fold, k_fold_cross_validation,
                                            vid_files, labels)
            )

    # Read in videos and transform to 3D
    print()
    print("===========================")
    print("Reading in videos")
    print("===========================")
    vid3d = Videoto3D(args.videos, width=args.width, height=args.height, depth=args.depth,
                      framerate=args.frame_rate, grayscale=args.grayscale, optical_flow_type=args.optical_flow_type)
    X_train, train_labels_text, train_files = vid3d.video3d(
        train_files,
        train_labels,
        save=None
    )
    X_validation, validation_labels_text, validation_files = vid3d.video3d(
        validation_files,
        validation_labels,
        save=None
    )
    X_test, test_labels_text, test_files = vid3d.video3d(
        test_files,
        test_labels,
        save=None
    )

    # One-hot encoding
    lb = LabelBinarizer()
    lb.fit(train_labels_text)
    Y_train = lb.transform(train_labels_text)
    Y_validation = np.array(lb.transform(validation_labels_text))
    Y_test = np.array(lb.transform(test_labels_text))

    # Model genesis requires different dataset shape than other cnns.
    if args.architecture == "model_genesis":
        # Rearrange to put channels first and depth last
        X_train = np.transpose(X_train, [0, 4, 2, 3, 1])
        X_validation = np.transpose(X_validation, [0, 4, 2, 3, 1])
        X_test = np.transpose(X_test, [0, 4, 2, 3, 1])

        # Repeat frames since depth of model is 32
        required_depth = 32
        num_repeats = required_depth // args.depth
        extra = required_depth - args.depth * num_repeats
        repeat_list = [num_repeats for _ in range(args.depth)]
        for i in range(extra):
            repeat_list[i] += 1
        print(f"With depth = {args.depth} and required_depth = {required_depth}, will repeat frames like so " +
              f"{repeat_list} so the new depth is {sum(repeat_list)}")
        X_train = np.repeat(X_train, repeat_list, axis=-1)
        X_validation = np.repeat(X_validation, repeat_list, axis=-1)
        X_test = np.repeat(X_test, repeat_list, axis=-1)

    input_shape = X_train.shape[1:]
    print(f"input_shape = {input_shape}")

    generator = DataGenerator(X_train, Y_train, args.batch_size, input_shape, lb.classes_, shuffle=True)

    # VISUALIZE
    if args.visualize:
        num_show = 8
        print(f"Visualizing {num_show} video clips")
        for i in range(X_train.shape[0]):
            # End early
            if i >= num_show:
                break

            video_clip = X_train[i]
            label = Y_train[i]
            for j in range(video_clip.shape[0]):
                frame = video_clip[j]
                num_channels = frame.shape[2]
                if num_channels == 1 or num_channels == 3:
                    cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Example-{i}_Frame-{j}_Label-{label}.jpg"), 255*frame)
                elif num_channels == 6:
                    rgb_frame = frame[:, :, :3]
                    optical_flow_frame = frame[:, :, 3:]
                    cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Example-{i}_Frame-{j}_Label-{label}.jpg"),
                                255*rgb_frame)
                    cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Example-{i}_Frame-{j}_Label-{label}-opt.jpg"),
                                255*optical_flow_frame)

        print("Visualizing 1 batch of augmented video clips")
        batchX, batchY = generator[0]
        for i, (video_clip, label) in enumerate(zip(batchX, batchY)):
            for j, frame in enumerate(video_clip):
                num_channels = frame.shape[2]
                if num_channels == 1 or num_channels == 3:
                    cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Augment-Example-{i}_Frame-{j}_Label-{label}.jpg"),
                                255*frame)
                elif num_channels == 6:
                    rgb_frame = frame[:, :, :3]
                    optical_flow_frame = frame[:, :, 3:]
                    cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Augment-Example-{i}_Frame-{j}_Label-{label}.jpg"),
                                255*rgb_frame)
                    cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Augment-Example-{i}_Frame-{j}_Label-{label}-opt.jpg"),
                                255*optical_flow_frame)

    print()
    print("===========================")
    print("Printing details about dataset")
    print("===========================")
    print(f"X_train.shape, Y_train.shape = {X_train.shape}, {Y_train.shape}")
    print(f"X_validation.shape, Y_validation.shape = {X_validation.shape}, {Y_validation.shape}")
    print(f"X_test.shape, Y_test.shape = {X_test.shape}, {Y_test.shape}")
    nb_classes = len(np.unique(train_labels_text))
    print(f"nb_classes, np.max(X_train) = {nb_classes}, {np.max(X_train)}")
    train_uniques, train_counts = np.unique(train_labels_text, return_counts=True)
    validation_uniques, validation_counts = np.unique(validation_labels_text, return_counts=True)
    test_uniques, test_counts = np.unique(test_labels_text, return_counts=True)
    print("unique labels in train", (train_uniques, train_counts))
    print("unique labels in validation", (validation_uniques, validation_counts))
    print("unique labels in test", (test_uniques, test_counts))

    class_weight = {i: sum(train_counts) / train_counts[i] for i in range(len(train_counts))}
    print(f"class_weight = {class_weight}")

    model = VIDEO_MODEL_FACTORY[args.architecture](input_shape, nb_classes, args.pretrained_cnn)

    tf.keras.utils.plot_model(model, os.path.join(FINAL_OUTPUT_DIR, f"{args.architecture}.png"), show_shapes=True)

    if args.optimizer == "adam":
        opt = Adam(lr=args.learning_rate)
    else:
        print(f"WARNING: invalid optimizer {args.optimizer}")

    model.compile(
        optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy']
    )

    wandb.init(entity='tylerlum', project=args.wandb_project)
    wandb.config.update(args)
    wandb.config.final_output_dir = FINAL_OUTPUT_DIR

    callbacks = []
    if args.reduce_learning_rate:
        reduce_learning_rate_loss = ReduceLROnPlateau(
            monitor=args.reduce_learning_rate_monitor,
            factor=args.reduce_learning_rate_factor,
            patience=args.reduce_learning_rate_patience,
            mode=args.reduce_learning_rate_mode,
            verbose=1,
            epsilon=1e-4,
        )
        callbacks.append(reduce_learning_rate_loss)
    if args.confusion_matrix_each_epoch:
        callbacks.append(ConfusionMatrixEachEpochCallback(X_validation, Y_validation, lb.classes_))

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
            shuffle=True,
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
            shuffle=True,
            class_weight=class_weight,
            callbacks=callbacks,
        )

    print()
    print("===========================")
    print("Evaluating network...")
    print("===========================")
    # Running inference on training set can cause out of memory issue when using larger framerate (OK on DGX)
    trainLoss, trainAcc = model.evaluate(X_train, Y_train, verbose=1)
    print('train loss:', trainLoss)
    print('train accuracy:', trainAcc)
    validationLoss, validationAcc = model.evaluate(X_validation, Y_validation, verbose=1)
    print('Validation loss:', validationLoss)
    print('Validation accuracy:', validationAcc)
    testLoss, testAcc = model.evaluate(X_test, Y_test, verbose=1)
    print('Test loss:', testLoss)
    print('Test accuracy:', testAcc)

    rawTrainPredIdxs = model.predict(X_train, batch_size=args.batch_size)
    rawValidationPredIdxs = model.predict(X_validation, batch_size=args.batch_size)
    rawTestPredIdxs = model.predict(X_test, batch_size=args.batch_size)

    def savePredictionsToCSV(rawPredIdxs, csvFilename, directory=FINAL_OUTPUT_DIR):
        df = pd.DataFrame(rawPredIdxs)
        df.to_csv(os.path.join(directory, csvFilename))
    savePredictionsToCSV(rawTrainPredIdxs, "train_preds_last_epoch.csv")
    savePredictionsToCSV(rawValidationPredIdxs, "validation_preds_last_epoch.csv")
    savePredictionsToCSV(rawTestPredIdxs, "test_preds_last_epoch.csv")

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    trainPredIdxs = np.argmax(rawTrainPredIdxs, axis=1)
    validationPredIdxs = np.argmax(rawValidationPredIdxs, axis=1)
    testPredIdxs = np.argmax(rawTestPredIdxs, axis=1)

    trainTrueIdxs = np.argmax(Y_train, axis=1)
    validationTrueIdxs = np.argmax(Y_validation, axis=1)
    testTrueIdxs = np.argmax(Y_test, axis=1)

    classes_with_validation = [f"{c} Validation" for c in lb.classes_]
    wandb.sklearn.plot_classifier(model, X_train, X_validation, trainTrueIdxs, validationTrueIdxs, validationPredIdxs,
                                  rawValidationPredIdxs, classes_with_validation, model_name=f"{args.architecture}")
    classes_with_test = [f"{c} Test" for c in lb.classes_]
    wandb.log({'Test Confusion Matrix': wandb.plots.HeatMap(classes_with_test, classes_with_test,
                                                            matrix_values=confusion_matrix(testTrueIdxs, testPredIdxs),
                                                            show_text=True)})

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    def printAndSaveClassificationReport(trueIdxs, predIdxs, classes, reportFilename, directory=FINAL_OUTPUT_DIR):
        print(f'classification report sklearn for {reportFilename}')
        print(
            classification_report(
                trueIdxs, predIdxs, target_names=classes
            )
        )

        report = classification_report(
            trueIdxs, predIdxs, target_names=classes, output_dict=True
        )
        reportDf = pd.DataFrame(report).transpose()
        reportDf.to_csv(os.path.join(directory, reportFilename))

        wandb_log_classification_table_and_plots(report, reportFilename)

    printAndSaveClassificationReport(trainTrueIdxs, trainPredIdxs, lb.classes_, "trainReport.csv")
    printAndSaveClassificationReport(validationTrueIdxs, validationPredIdxs, lb.classes_, "validationReport.csv")
    printAndSaveClassificationReport(testTrueIdxs, testPredIdxs, lb.classes_, "testReport.csv")

    def printAndSaveConfusionMatrix(trueIdxs, predIdxs, classes, confusionMatrixFilename, directory=FINAL_OUTPUT_DIR):
        print(f'confusion matrix for {confusionMatrixFilename}')

        cm = confusion_matrix(trueIdxs, predIdxs)
        # show the confusion matrix, accuracy, sensitivity, and specificity
        print(cm)

        plt.figure()
        cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        cmDisplay.plot()
        plt.savefig(os.path.join(directory, confusionMatrixFilename))
    printAndSaveConfusionMatrix(trainTrueIdxs, trainPredIdxs, lb.classes_, "trainConfusionMatrix.png")
    printAndSaveConfusionMatrix(validationTrueIdxs, validationPredIdxs, lb.classes_, "validationConfusionMatrix.png")
    printAndSaveConfusionMatrix(testTrueIdxs, testPredIdxs, lb.classes_, "testConfusionMatrix.png")

    if args.save_model:
        print(f'Saving COVID-19 detector model on {FINAL_OUTPUT_DIR} data...')
        model.save(os.path.join(FINAL_OUTPUT_DIR, 'last_epoch'), save_format='h5')

    def calculate_patient_wise(files, x, y, model):
        # Calculate mean of video clips to predict patient-wise classification
        gt = []
        preds = []
        files = np.array(files)
        for video in np.unique(files):
            current_data = x[files == video]
            current_labels = y[files == video]
            true_label = np.argmax(current_labels[0])
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


if __name__ == '__main__':
    main()
