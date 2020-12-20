seed_value = 1233
import wandb
from wandb.keras import WandbCallback
import itertools
import argparse
import math
import json
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import sys
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
import imgaug
imgaug.random.seed(seed_value)
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
from pocovidnet.video_augmentation import DataGenerator

from pocovidnet import VIDEO_MODEL_FACTORY
from pocovidnet.videoto3d import Videoto3D
from pocovidnet.wandb import WandbClassificationCallback, wandb_log_classification_report, wandb_log_classification_table_and_plots
from datetime import datetime
from datetime import date


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
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--fr', type=int, default=5)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--model_id', type=str, default="2D_CNN_average")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--trainable_base_layers', type=int, default=0)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--wandb_project', type=str, default="covid-video-debugging")
    parser.add_argument('--reduce_lr', action='store_true')
    parser.add_argument(
        '--weight_path', type=str, default='../Genesis_Chest_CT.h5'
    )

    args = parser.parse_args()
    print(args)

    # Out model directory
    MODEL_D = args.output
    if not os.path.isdir(MODEL_D):
        if not os.path.exists(MODEL_D):
            os.makedirs(args.output)

    SAVE_DIR = '../data/video_input_data/'
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    train_save_path, validation_save_path, test_save_path = (os.path.join(SAVE_DIR, "conv3d_train_fold_" + str(args.fold) + ".dat"),
                                                             os.path.join(SAVE_DIR, "conv3d_validation_fold_" + str(args.fold) + ".dat"),
                                                             os.path.join(SAVE_DIR, "conv3d_test_fold_" + str(args.fold) + ".dat"))

    FINAL_OUTPUT_DIR = os.path.join(MODEL_D, datestring)
    if not os.path.exists(FINAL_OUTPUT_DIR):
        os.makedirs(FINAL_OUTPUT_DIR)

    # Load saved data or read in videos
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
            vid_files, labels, stratify=labels, test_size=0.2, random_state=seed_value
        )
        train_files, validation_files, train_labels, validation_labels = train_test_split(
            train_files, train_labels, stratify=train_labels, test_size=0.2, random_state=seed_value
        )
        full_video_train_files, full_video_validation_files, full_video_test_files, full_video_train_labels, full_video_validation_labels, full_video_test_labels = train_files, validation_files, test_files, train_labels, validation_labels, test_labels

        # Read in videos and transform to 3D
        vid3d = Videoto3D(args.videos, width=args.width, height=args.height, depth=args.depth, framerate=args.fr)

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

    input_shape = X_train.shape[1:]
    print(f"input_shape = {input_shape}")
    if args.depth != input_shape[0] or args.height != input_shape[1] or args.width != input_shape[2]:
        print("WARNING: Loaded shape is not same as desired shape")
        print("args.depth != input_shape[1] or args.height != input_shape[2] or args.width != input_shape[3]")
        print(f"{args.depth} != {input_shape[1]} or {args.height} != {input_shape[2]} or {args.width} != {input_shape[3]}")
        print(f"{args.depth != input_shape[1]} or {args.height != input_shape[2]} or {args.width != input_shape[3]}")

    generator = DataGenerator(X_train, Y_train, args.batch, input_shape, lb.classes_, shuffle=False)

    ## VISUALIZE
    if args.visualize:
#        for i in range(20):
#            # 'video' should be either a list of images from type of numpy array or PIL images
#            orig_video = X_train[i]
#            video = np.squeeze(orig_video)
#            video = [video, video, video]
#            video = np.stack(video, axis=3)
#            print(video.shape)
#            video_aug = seq(video*255)
#            print(len(video_aug))
#            print(video_aug[0].shape)
#            for j in range(1):
#                import cv2
#                print(f"Frame {j}")
#                frame = video_aug[j]
#                print(f"np.max(frame) = {np.max(frame)}")
#                print(f"np.min(frame) = {np.min(frame)}")
#                cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"video_aug-{i}_Frame-{j}.jpg"), frame)
#                cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"video_aug-{i}_Frame-{j}org.jpg"), 255*orig_video[j])
#
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
            if i > 8:
                break

        batchX, batchY = generator[0]
        i = 0
        for frames, label in zip(batchX, batchY):
            j = 0
            for frame in frames:
                import cv2
                cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Augmented-Example-{i}_Frame-{j}_Label-{label}.jpg"), 255*frame)
                j += 1
            i += 1


    # Verbose
    print("testing on split", args.fold)
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

    model = VIDEO_MODEL_FACTORY[args.model_id](input_shape, nb_classes)

    tf.keras.utils.plot_model(model, os.path.join(FINAL_OUTPUT_DIR, f"{args.model_id}.png"), show_shapes=True)

    opt = Adam(lr=args.lr)
    model.compile(
        optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy']
    )

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
    log_dir = os.path.join(FINAL_OUTPUT_DIR, "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    file_writer_cm = tf.summary.create_file_writer(os.path.join(log_dir, "confusion_matrix"))
    def plot_confusion_matrix(cm, class_names):
      """
      Returns a matplotlib figure containing the plotted confusion matrix.

      Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
      """
      figure = plt.figure(figsize=(8, 8))
      plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
      plt.title("Confusion matrix")
      plt.colorbar()
      tick_marks = np.arange(len(class_names))
      plt.xticks(tick_marks, class_names, rotation=45)
      plt.yticks(tick_marks, class_names)

      # Compute the labels from the normalized confusion matrix.
      labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

      # Use white text if squares are dark; otherwise black.
      threshold = cm.max() / 2.
      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

      plt.tight_layout()
      plt.ylabel('True label')
      plt.xlabel('Predicted label')
      return figure
    def plot_to_image(figure):
      """Converts the matplotlib plot specified by 'figure' to a PNG image and
      returns it. The supplied figure is closed and inaccessible after this call."""
      # Save the plot to a PNG in memory.
      import io
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      # Closing the figure prevents it from being displayed directly inside
      # the notebook.
      plt.close(figure)
      buf.seek(0)
      # Convert PNG buffer to TF image
      image = tf.image.decode_png(buf.getvalue(), channels=4)
      # Add the batch dimension
      image = tf.expand_dims(image, 0)
      return image
    def log_confusion_matrix(epoch, logs):
      # Use the model to predict the values from the validation dataset.
      def get_cm_image(X, Y):
          test_pred_raw = model.predict(X)
          test_pred = np.argmax(test_pred_raw, axis=1)

          # Calculate the confusion matrix.
          cm = confusion_matrix(np.argmax(Y, axis=1), test_pred)
          # Log the confusion matrix as an image summary.
          figure = plot_confusion_matrix(cm, class_names=lb.classes_)
          cm_image = plot_to_image(figure)
          return cm_image

      validation_cm = get_cm_image(X_validation, Y_validation)
      test_cm = get_cm_image(X_test, Y_test)
      # Log the confusion matrix as an image summary.
      with file_writer_cm.as_default():
        tf.summary.image("Validation Confusion Matrix", validation_cm, step=epoch)
        tf.summary.image("Test Confusion Matrix", test_cm, step=epoch)

    # Define the per-epoch callback
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
    wandb.init(entity='tylerlum', project=args.wandb_project)
    config = wandb.config
    config.learning_rate = args.lr
    config.batch_size = args.batch
    config.activation = 'relu'
    config.optimizer = 'adam'
    config.epochs = args.epoch
    config.architecture = args.model_id
    config.frame_rate = args.fr
    config.depth = args.depth
    config.width = args.width
    config.height = args.height
    config.output_dir = FINAL_OUTPUT_DIR
    config.augment = args.augment
    if args.reduce_lr:
        config.reduce_lr_monitor = reduce_lr_loss.monitor
        config.reduce_lr_factor = reduce_lr_loss.factor
        config.reduce_lr_patience = reduce_lr_loss.patience
        config.reduce_lr_mode = reduce_lr_loss.mode

    wandb_callback = WandbClassificationCallback(log_confusion_matrix=True, confusion_classes=len(lb.classes_), validation_data=(X_validation, Y_validation), labels=lb.classes_)

    callbacks = [wandb_callback]
    if args.reduce_lr:
        callbacks.append(reduce_lr_loss)

    print("ABOUT TO TRAIN THIS MODEL")
    print(model.summary())

    if args.augment:
        H = model.fit(
            generator,
            validation_data=(X_validation, Y_validation),
            epochs=args.epoch,
            batch_size=args.batch,
            verbose=1,
            shuffle=False,
            class_weight=class_weight,
            # use_multiprocessing=True,
            # workers=2,  # Empirically best performance
            # callbacks=[earlyStopping, reduce_lr_loss, tensorboard_callback, cm_callback, WandbClassificationCallback(log_confusion_matrix=True, confusion_classes=len(lb.classes_), validation_data=(X_validation, Y_validation), labels=lb.classes_)],
            callbacks=[WandbClassificationCallback(log_confusion_matrix=True, confusion_classes=len(lb.classes_), validation_data=(X_validation, Y_validation), labels=lb.classes_)],
        )
    else:
        H = model.fit(
            X_train, Y_train,
            validation_data=(X_validation, Y_validation),
            epochs=args.epoch,
            batch_size=args.batch,
            verbose=1,
            shuffle=False,
            class_weight=class_weight,
            # use_multiprocessing=True,
            # workers=2,  # Empirically best performance
            callbacks=[WandbClassificationCallback(log_confusion_matrix=True, confusion_classes=len(lb.classes_), validation_data=(X_validation, Y_validation), labels=lb.classes_)],
        )


    print('Evaluating network...')
    trainLoss, trainAcc = model.evaluate(X_train, Y_train, verbose=1)
    print('train loss:', trainLoss)
    print('train accuracy:', trainAcc)
    validationLoss, validationAcc = model.evaluate(X_validation, Y_validation, verbose=1)
    print('Validation loss:', validationLoss)
    print('Validation accuracy:', validationAcc)
    testLoss, testAcc = model.evaluate(X_test, Y_test, verbose=1)
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

if __name__ == '__main__':
    main()
