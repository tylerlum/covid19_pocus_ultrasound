import wandb
from wandb.keras import WandbCallback
import argparse
import os
from tqdm import tqdm
import random
import imgaug
import warnings
import math
from keras import backend as K

import tensorflow_addons as tfa

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold, KFold
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

from pocovidnet.video_augmentation import DataGenerator

from pocovidnet import VIDEO_MODEL_FACTORY, PRETRAINED_CNN_PREPROCESS_FACTORY
from pocovidnet.videoto3d import Videoto3D
from pocovidnet.wandb import ConfusionMatrixEachEpochCallback, wandb_log_classification_table_and_plots
from pocovidnet.read_mat import loadmat
from datetime import datetime
from datetime import date
from keras.layers import Dropout
from keras.models import Model

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
    parser.add_argument('--wandb_project', type=str, default="optimize_multihead", help='wandb project name')

    # Input files
    parser.add_argument(
        '--videos',
        type=str,
        default='../data/pocus_videos_Jan_30_2021/convex',
        help='directory where videos are stored'
    )
    parser.add_argument('--mat', type=str2bool, nargs='?', const=True, default=False,
                        help='for pocovidnet dataset, do not use. Only used for reading mat files, pass in the lung/labelled folder to --videos')
    parser.add_argument('--multitask', type=str2bool, nargs='?', const=True, default=True,
                        help='do multitask network or not')

    # Output files
    parser.add_argument('--save_model', type=str2bool, nargs='?', const=True, default=False, help='save final model')
    parser.add_argument('--visualize', type=str2bool, nargs='?', const=True, default=False,
                        help='Save images to visualize in output dir')

    # Uncertainty plotting and calculation
    parser.add_argument('--uncertainty', type=str2bool, nargs='?', const=True, default=False,
                        help='calculate and plot uncertainty')

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

    if args.architecture.endswith('evidential'):
        evidential = True
        print("This model uses evidential")
    else:
        evidential = False
        print("This model does not use evidential")

    if args.multitask == True:
        print('this is a multitask network')
    else:
        print('this is a regular network, not a multitask one')

    # Deterministic behavior
    set_random_seed(args.random_seed)

    # Output directory
    OUTPUT_DIR = "multihead_model_outputs"
    if not os.path.isdir(OUTPUT_DIR):
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
    FINAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, datestring)
    if not os.path.exists(FINAL_OUTPUT_DIR):
        os.makedirs(FINAL_OUTPUT_DIR)

    SAVE_DIR = '../data/video_input_data/'
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Setup folds

    train_true = [[], []]
    train_pred = [[], []]
    val_true = [[], []]
    val_pred = [[], []]
    test_true = [[], []]
    test_pred = [[], []]

    k_fold_cross_validation = KFold(n_splits=args.num_folds, random_state=args.random_seed, shuffle=True)

    for test_fold in range(args.num_folds):
        validation_fold = (test_fold + 1) % args.num_folds  # Select validation fold
        print()
        print("===========================")
        print(f"Performing k-fold splitting with validation fold {validation_fold} and test fold {test_fold}")
        print("===========================")

        # k_fold_cross_validation = StratifiedKFold(n_splits=args.num_folds, random_state=args.random_seed,
        #                                           shuffle=True)  # Doesn't work when not enough datapoints of each class

        def get_train_validation_test_split(validation_fold, test_fold, k_fold_cross_validation, vid_files, labels):
            for i, (train_index, test_index) in enumerate(k_fold_cross_validation.split(vid_files, labels)):
                if i == validation_fold:
                    validation_indices = test_index
                elif i == test_fold:
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

        # Use pocovid dataset
        if not args.mat:
            # Get videos and labels
            class_short = ["cov", "pne", "reg"]  # TODO, don't load pne this time see what happens
            # class_short = ["cov", "reg"]
            vid_files = [
                v for v in os.listdir(args.videos) if v[:3].lower() in class_short
            ]
            labels = [vid[:3].lower() for vid in vid_files]

            # Read in videos and transform to 3D
            print()
            print("===========================")
            print("Reading in videos, not from mat files")
            print("===========================")

            train_files, train_labels, validation_files, validation_labels, test_files, test_labels = (
                get_train_validation_test_split(validation_fold, test_fold, k_fold_cross_validation, vid_files, labels)
            )
            vid3d = Videoto3D(args.videos, width=args.width, height=args.height, depth=args.depth,
                              framerate=args.frame_rate, grayscale=args.grayscale,
                              optical_flow_type=args.optical_flow_type,
                              pretrained_cnn=args.pretrained_cnn)
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

        # Use private lung dataset
        else:
            # Split up mat files to train/validation/test
            all_patient_dirs = [os.path.join(args.videos, name) for name in os.listdir(args.videos)
                                if os.path.isdir(os.path.join(args.videos, name))]
            # Get all mat files
            all_mat_files = []
            for patient_dir in all_patient_dirs:
                for mat_or_dir in os.listdir(patient_dir):
                    path_to_mat_or_dir = os.path.join(patient_dir, mat_or_dir)

                    # Handle folders with mat files one level deeper
                    if os.path.isdir(path_to_mat_or_dir):
                        for mat in os.listdir(path_to_mat_or_dir):
                            all_mat_files.append(os.path.join(path_to_mat_or_dir, mat))
                    else:
                        all_mat_files.append(path_to_mat_or_dir)

            def get_labels(mat_files):
                labels = []
                print("Getting labels for stratified k-fold splitting")
                for mat_file in tqdm(mat_files):
                    mat = loadmat(mat_file)

                    # Get labels
                    b_lines = mat['labels']['B-lines']
                    stop_frame = mat['labels']['stop_frame']
                    start_frame = mat['labels']['start_frame']
                    subpleural_consolidations = mat['labels']['Sub-pleural consolidations']
                    pleural_irregularities = mat['labels']['Pleural irregularities']
                    a_lines = mat['labels']['A-lines']
                    lobar_consolidations = mat['labels']['Lobar consolidations']
                    pleural_effusions = mat['labels']['Pleural effussions']
                    no_lung_sliding = mat['labels']['No lung sliding']

                    b = 0  # binary b_line label, for now
                    if b_lines >= 1:
                        b = 1
                    labels.append({'head_0': b,
                                   'head_1': a_lines})

                return labels

            all_labels = get_labels(all_mat_files)

            train_mats, train_labels, validation_mats, validation_labels, test_mats, test_labels = (
                get_train_validation_test_split(validation_fold, test_fold, k_fold_cross_validation, all_mat_files,
                                                all_labels)
            )

            def get_video_clips_and_labels(mat_files):
                video_clips = []
                labels = []

                # Mat files
                print("Collecting video clips and labels")
                for mat_file in tqdm(mat_files):
                    mat = loadmat(os.path.join(patient_dir, mat_file))

                    # Get labels
                    b_lines = mat['labels']['B-lines']
                    stop_frame = mat['labels']['stop_frame']
                    start_frame = mat['labels']['start_frame']
                    subpleural_consolidations = mat['labels']['Sub-pleural consolidations']
                    pleural_irregularities = mat['labels']['Pleural irregularities']
                    a_lines = mat['labels']['A-lines']
                    lobar_consolidations = mat['labels']['Lobar consolidations']
                    pleural_effusions = mat['labels']['Pleural effussions']
                    no_lung_sliding = mat['labels']['No lung sliding']

                    # Calculate frequency of getting frames
                    # Some mat files are missing FrameTime
                    if 'FrameTime' not in mat['dicom_info']:
                        time_between_frames_ms = 50  # Typically is 50 for linear and 30 for curved
                    else:
                        time_between_frames_ms = mat['dicom_info']['FrameTime']
                    video_framerate = 1.0 / (time_between_frames_ms / 1000)
                    show_every = math.ceil(video_framerate / args.frame_rate)

                    # Get cine
                    if 'cropped' in mat.keys() and 'cleaned' in mat.keys():
                        raise ValueError(f"{mat_file} has both cropped and cleaned")
                    if 'cropped' not in mat.keys() and 'cleaned' not in mat.keys():
                        raise ValueError(f"{mat_file} has neither cropped or cleaned")
                    if 'cropped' in mat.keys():
                        cine = mat['cropped']
                    if 'cleaned' in mat.keys():
                        cine = mat['cleaned']

                    # start_frame = 0
                    # stop_frame = cine.shape[2]

                    num_video_frames = (stop_frame - start_frame + 1) // show_every
                    num_clips = num_video_frames // args.depth

                    # Video clips
                    for i in range(num_clips):
                        start, stop = start_frame + i * args.depth * show_every, start_frame + (
                                i + 1) * args.depth * show_every
                        clip_data = cine[:, :, start:stop:show_every]
                        video_clip = []

                        # Frames
                        for frame_i in range(clip_data.shape[-1]):
                            frame = cv2.resize(clip_data[:, :, frame_i], (args.width, args.height))
                            frame = frame[:, :, np.newaxis]
                            frame = cv2.merge([frame, frame, frame])
                            video_clip.append(frame)

                        video_clips.append(video_clip)

                        b = [1, 0]  # binary b_line label, for now
                        if b_lines >= 1:
                            b = [0, 1]
                        if a_lines == 1:
                            a = [0, 1]
                        else:
                            a = [1, 0]
                        # note: omitting the no_lung_sliding label because of the
                        # low numbers (5 in total cases) that all appeared in test set
                        labels.append({'head_0': b,
                                       'head_1': a})
                        # if args.mat_task == 'a_lines':
                        #     labels.append(a_lines)
                        # elif args.mat_task == 'b_lines_binary':
                        #     labels.append(1 if b_lines > 0 else 0)
                        # elif args.mat_task == 'b_lines':
                        #     labels.append(b_lines)

                X = np.array(video_clips)
                Y = np.array(labels)
                return X, Y

            # Get video clips and labels
            X_train, Y_train = get_video_clips_and_labels(train_mats)
            X_validation, Y_validation = get_video_clips_and_labels(validation_mats)
            X_test, Y_test = get_video_clips_and_labels(test_mats)

            lb = LabelBinarizer()
            # Onehot encode labels
            if not args.multitask:
                train_labels_text = Y_train
                validation_labels_text = Y_validation
                test_labels_text = Y_test
                lb.fit(train_labels_text)
                Y_train = np.array(lb.transform(train_labels_text))
                Y_validation = np.array(lb.transform(Y_validation))
                Y_test = np.array(lb.transform(Y_test))
            else:
                Y_train = np.array(Y_train)
                Y_validation = np.array(Y_validation)
                Y_test = np.array(Y_test)

            # Workaround to get text class names
            # lb.classes_ = ['b_lines', 'subpleural_consolidations',
            #                'a_lines', 'pleural_irregularities', 'lobar_consolidations',
            #                'pleural_effusions']
            lb.classes_ = ['binary_b_line', 'a_line']

        input_shape = X_train.shape[1:]
        print(f"input_shape = {input_shape}")

        # VISUALIZE
        if args.visualize:
            num_show = 100
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
                        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Example-{i}_Frame-{j}_Label-{label}.jpg"),
                                    255 * frame)
                    elif num_channels == 6:
                        rgb_frame = frame[:, :, :3]
                        optical_flow_frame = frame[:, :, 3:]
                        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Example-{i}_Frame-{j}_Label-{label}.jpg"),
                                    255 * rgb_frame)
                        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Example-{i}_Frame-{j}_Label-{label}-opt.jpg"),
                                    255 * optical_flow_frame)

            print("Visualizing 1 batch of augmented video clips")
            batchX, batchY = generator[0]
            for i, (video_clip, label) in enumerate(zip(batchX, batchY)):
                for j, frame in enumerate(video_clip):
                    num_channels = frame.shape[2]
                    if num_channels == 1 or num_channels == 3:
                        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Augment-Example-{i}_Frame-{j}_Label-{label}.jpg"),
                                    255 * frame)
                    elif num_channels == 6:
                        rgb_frame = frame[:, :, :3]
                        optical_flow_frame = frame[:, :, 3:]
                        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Augment-Example-{i}_Frame-{j}_Label-{label}.jpg"),
                                    255 * rgb_frame)
                        cv2.imwrite(
                            os.path.join(FINAL_OUTPUT_DIR, f"Augment-Example-{i}_Frame-{j}_Label-{label}-opt.jpg"),
                            255 * optical_flow_frame)

        # if not args.mat:
        #     print()
        #     print("===========================")
        #     print("Removing pneumonia classes")
        #     print("===========================")
        #     train_pne_idx = np.where(np.argmax(Y_train, axis=1) == 1)
        #     X_train = np.delete(X_train, train_pne_idx, axis=0)
        #     Y_train = np.delete(Y_train, train_pne_idx, axis=0)
        #     train_labels_text = np.delete(train_labels_text, train_pne_idx, axis=0)
        #     Y_train = np.delete(Y_train, 1, 1)
        #
        #     test_pne_idx = np.where(np.argmax(Y_test, axis=1) == 1)
        #     X_test = np.delete(X_test, test_pne_idx, axis=0)
        #     Y_test = np.delete(Y_test, test_pne_idx, axis=0)
        #     test_labels_text = np.delete(test_labels_text, test_pne_idx, axis=0)
        #     Y_test = np.delete(Y_test, 1, 1)
        #
        #     val_pne_idx = np.where(np.argmax(Y_validation, axis=1) == 1)
        #     X_validation = np.delete(X_validation, val_pne_idx, axis=0)
        #     Y_validation = np.delete(Y_validation, val_pne_idx, axis=0)
        #     validation_labels_text = np.delete(validation_labels_text, val_pne_idx, axis=0)
        #     Y_validation = np.delete(Y_validation, 1, 1)
        #
        #     lb.classes_ = ['cov', 'reg']

        print()
        print("===========================")
        print("Printing details about dataset")
        print("===========================")
        print(f"X_train.shape, Y_train.shape = {X_train.shape}, {Y_train.shape}")
        print(f"X_validation.shape, Y_validation.shape = {X_validation.shape}, {Y_validation.shape}")
        print(f"X_test.shape, Y_test.shape = {X_test.shape}, {Y_test.shape}")
        if not args.multitask:
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
        else:
            nb_classes = 2
            print("num of classes: ", nb_classes)

        generator = DataGenerator(X_train, Y_train, args.batch_size, input_shape, shuffle=True)

        # tf.keras.utils.plot_model(model, os.path.join(FINAL_OUTPUT_DIR, f"{args.architecture}.png"), show_shapes=True)

        # evidential loss function
        def KL(alpha, K):
            beta = tf.constant(np.ones((1, K)), dtype=tf.float32)
            S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)

            KL = tf.reduce_sum((alpha - beta) * (tf.math.digamma(alpha) - tf.math.digamma(S_alpha)), axis=1,
                               keepdims=True) + \
                 tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True) + \
                 tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True) - tf.math.lgamma(
                tf.reduce_sum(beta, axis=1, keepdims=True))
            return KL

        def loss_eq5(actual, pred, K, global_step, annealing_step):
            p = actual
            p = tf.dtypes.cast(p, tf.float32)
            alpha = pred + 1.
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            loglikelihood = tf.reduce_sum((p - (alpha / S)) ** 2, axis=1, keepdims=True) + tf.reduce_sum(
                alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True)
            KL_reg = tf.minimum(1.0, tf.cast(global_step / annealing_step, tf.float32)) * KL((alpha - 1) * (1 - p) + 1,
                                                                                             K)
            return loglikelihood + KL_reg

        ev_loss = (
            lambda actual, pred: tf.reduce_mean(loss_eq5(actual, pred, nb_classes, 1, 100))
        )

        loss = categorical_crossentropy
        if evidential:
            loss = ev_loss

        if args.optimizer == "adam":
            opt = Adam(lr=args.learning_rate)
        else:
            print(f"WARNING: invalid optimizer {args.optimizer}")

        # source: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
        def weighted_categorical_crossentropy(weights):
            """
            A weighted version of keras.objectives.categorical_crossentropy

            Variables:
                weights: numpy array of shape (C,) where C is the number of classes

            Usage:
                weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
                loss = weighted_categorical_crossentropy(weights)
                model.compile(loss=loss,optimizer='adam')
            """

            weights = K.variable(weights)

            def loss(y_true, y_pred):
                # scale predictions so that the class probas of each sample sum to 1
                y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
                # clip to prevent NaN's and Inf's
                y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
                # calc
                loss = tf.cast(y_true, dtype=tf.float32) * K.log(y_pred) * weights
                loss = -K.sum(loss, -1)
                return loss

            return loss

        # compiling model and or copying weights from one to another
        if args.multitask:

            gt_0 = np.array([t["head_{}".format(0)] for t in Y_train])
            gt_1 = np.array([t["head_{}".format(1)] for t in Y_train])

            gt_0 = np.argmax(gt_0, axis=1)
            gt_1 = np.argmax(gt_1, axis=1)

            n_samples_0 = gt_0.shape[0]
            n_samples_1 = gt_1.shape[0]

            weights_0 = [n_samples_0/(2*(n_samples_0 - np.sum(gt_0))), n_samples_0/(2*np.sum(gt_0))]
            weights_1 = [n_samples_1/(2*(n_samples_1 - np.sum(gt_1))), n_samples_1/(2*np.sum(gt_1))]
            print(weights_1, weights_0)
            print("compiling the multihead network")
            from tensorflow.keras import metrics
            losses = {
                "head_0": weighted_categorical_crossentropy(np.array(weights_0)),
                "head_1": weighted_categorical_crossentropy(np.array(weights_1)),
            }
            metrs = {
                "head_0": 'accuracy',
                "head_1": 'accuracy',
            }
            model = VIDEO_MODEL_FACTORY[args.architecture + '_multihead'](input_shape, 2, args.pretrained_cnn)
            model.compile(optimizer=opt, loss=losses, metrics=metrs)

        else:
            model = VIDEO_MODEL_FACTORY[args.architecture + "_multihead"](input_shape, 2, args.pretrained_cnn)
            losses = {
                "head_0": [tfa.losses.SigmoidFocalCrossEntropy()],
                "head_1": [tfa.losses.SigmoidFocalCrossEntropy()],
            }
            metrs = {
                "head_0": 'AUC',
                "head_1": 'AUC',
            }
            model.compile(optimizer=opt, loss=losses, metrics=metrs)
            model.load_weights('transformer_conv1d_9')
            print('loading done')
            new_model = VIDEO_MODEL_FACTORY[args.architecture](input_shape, 3, args.pretrained_cnn)
            # now copying the weight Up until where it's applicable,
            # the 4 last layers which are the heads have to be popped from model
            # instead a  3-class classification head is added here

            new_model.compile(
                # tfa.losses.SigmoidFocalCrossEntropy()
                optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy']
            )
            for i in range(5):
                wk0 = model.layers[i].get_weights()
                new_model.layers[i].set_weights(wk0)
            print("model source was copied into model target")
            model = new_model

        wandb.init(entity='mobina', project=args.wandb_project)
        wandb.config.update(args)
        wandb.config.final_output_dir = FINAL_OUTPUT_DIR

        callbacks = [WandbCallback()]
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
        # es = tf.keras.callbacks.EarlyStopping(
        #     monitor=args.reduce_learning_rate_monitor,
        #     patience=args.reduce_learning_rate_patience,
        #     verbose=1,
        #     mode=args.reduce_learning_rate_mode,
        #     restore_best_weights=True,
        # )
        # callbacks.append(es)

        print()
        print("===========================")
        print("About to train model")
        print("===========================")
        print(model.summary())

        if not args.multitask:
            print("fitting normally with validation set {} and test set {}".format(validation_fold, test_fold))

            H = model.fit(
                generator,
                validation_data=(X_validation, Y_validation),
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=1,
                shuffle=True,
                class_weight=class_weight,
                # callbacks=callbacks,
            )
        elif args.multitask:
            print("fitting multihead")
            H = model.fit(
                X_train,
                y={"head_0": np.array([t["head_0"] for t in Y_train]),
                   "head_1": np.array([t["head_1"] for t in Y_train]),
                   },
                validation_data=(X_validation, {
                    "head_0": np.array([t["head_0"] for t in Y_validation]),
                    "head_1": np.array([t["head_1"] for t in Y_validation]),
                }),
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=1,
                shuffle=True,
                callbacks=callbacks
            )
        print()
        print("===========================")
        print("Evaluating network...")
        print("===========================")
        # Running inference on training set can cause out of memory issue when using larger framerate (OK on DGX)
        if args.mat:

            print("train set")
            Y_train_pred = model.predict(X_train)
            for i in range(len(Y_train_pred)):
                print("result of head_{}".format(i))
                gt = np.array([t["head_{}".format(i)] for t in Y_train])
                gt = np.argmax(gt, axis=1)
                raw = np.argmax(Y_train_pred[i], axis=1)
                print(gt.shape, raw.shape)
                cm = confusion_matrix(gt, raw)
                print(classification_report(gt, raw))
                # show the confusion matrix, accuracy, sensitivity, and specificity
                print(cm)
                train_true[i].append(gt)
                train_pred[i].append(raw)

            print("val set")
            Y_validation_pred = model.predict(X_validation)
            for i in range(len(Y_validation_pred)):
                print("result of head_{}".format(i))
                gt = np.array([t["head_{}".format(i)] for t in Y_validation])
                gt = np.argmax(gt, axis=1)
                raw = np.argmax(Y_validation_pred[i], axis=1)
                cm = confusion_matrix(gt, raw)
                print(classification_report(gt, raw))
                # show the confusion matrix, accuracy, sensitivity, and specificity
                print(cm)
                val_true[i].append(gt)
                val_pred[i].append(raw)
            print("Test set")
            Y_test_pred = model.predict(X_test)
            for i in range(len(Y_test_pred)):
                print("result of head_{}".format(i))
                gt = np.array([t["head_{}".format(i)] for t in Y_test])
                gt = np.argmax(gt, axis=1)
                raw = np.argmax(Y_test_pred[i], axis=1)
                cm = confusion_matrix(gt, raw)
                print(classification_report(gt, raw))
                # show the confusion matrix, accuracy, sensitivity, and specificity
                print(cm)
                test_true[i].append(gt)
                print(gt.shape, np.array(test_true[i]).shape)
                test_pred[i].append(raw)
            if args.save_model:
                print(f'Saving multihead model  fold {test_fold} on {FINAL_OUTPUT_DIR} ...')
                model.save_weights(os.path.join(FINAL_OUTPUT_DIR, 'multihead_best_fold_{}'.format(test_fold)))


        if not args.mat:
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

            cm = confusion_matrix(trainTrueIdxs, trainPredIdxs)
            print("train")
            print(cm)

            cm = confusion_matrix(validationTrueIdxs, validationPredIdxs)
            print("validation")
            print(cm)

            cm = confusion_matrix(testTrueIdxs, testPredIdxs)
            print("test")
            print(cm)

            train_true.append(trainTrueIdxs)
            train_pred.append(trainPredIdxs)
            val_true.append(validationTrueIdxs)
            val_pred.append(validationPredIdxs)
            test_true.append(testTrueIdxs)
            test_pred.append(testPredIdxs)

            def printAndSaveClassificationReport(trueIdxs, predIdxs, classes, reportFilename,
                                                 directory=FINAL_OUTPUT_DIR):
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

                # wandb_log_classification_table_and_plots(report, reportFilename)

            printAndSaveClassificationReport(trainTrueIdxs, trainPredIdxs, lb.classes_, "trainReport.csv")
            printAndSaveClassificationReport(validationTrueIdxs, validationPredIdxs, lb.classes_,
                                             "validationReport.csv")
            printAndSaveClassificationReport(testTrueIdxs, testPredIdxs, lb.classes_, "testReport.csv")

        def printAndSaveConfusionMatrix(trueIdxs, predIdxs, classes, confusionMatrixFilename,
                                        directory=FINAL_OUTPUT_DIR):
            print(f'confusion matrix for {confusionMatrixFilename}')

            cm = confusion_matrix(trueIdxs, predIdxs)
            # show the confusion matrix, accuracy, sensitivity, and specificity
            print(cm)

            plt.figure()
            cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            cmDisplay.plot()
            plt.savefig(os.path.join(directory, confusionMatrixFilename))

        # printAndSaveConfusionMatrix(trainTrueIdxs, trainPredIdxs, lb.classes_, "trainConfusionMatrix.png")
        # printAndSaveConfusionMatrix(validationTrueIdxs, validationPredIdxs, lb.classes_, "validationConfusionMatrix.png")
        # printAndSaveConfusionMatrix(testTrueIdxs, testPredIdxs, lb.classes_, "testConfusionMatrix.png")

        # if args.save_model:
        #     # print(f'Saving COVID-19 detector model on {FINAL_OUTPUT_DIR} data...')
        #     model.save_weights('multihead_best_fold_{}'.format(test_fold))

    # def calculate_patient_wise(files, x, y, model):
    #     # Calculate mean of video clips to predict patient-wise classification
    #     gt = []
    #     preds = []
    #     files = np.array(files)
    #     for video in np.unique(files):
    #         current_data = x[files == video]
    #         current_labels = y[files == video]
    #         true_label = np.argmax(current_labels[0])
    #         current_predictions = model.predict(current_data)
    #         prediction = np.argmax(np.mean(current_predictions, axis=0))
    #         gt.append(true_label)
    #         preds.append(prediction)
    #         print(f"video = {video}, true_label = {true_label}, prediction = {prediction}")
    #     return np.array(gt), np.array(preds)
    #
    # def create_mc_model(model, dropProb=0.5):
    #     layers = [layer for layer in model.layers]
    #     x = layers[0].output
    #     for i in range(1, len(layers)):
    #         # Replace dropout layers with MC dropout layers
    #         if isinstance(layers[i], Dropout):
    #             x = Dropout(dropProb)(x, training=True)
    #         else:
    #             x = layers[i](x)
    #     mc_model = Model(inputs=layers[0].input, outputs=x)
    #     mc_model.set_weights(model.get_weights())
    #     return mc_model
    #
    # def patient_wise_uncertainty(files, x, y, model):
    #     gt = []
    #     logits = []
    #     videos = []
    #     files = np.array(files)
    #     mc_model = create_mc_model(model)
    #     mc_uncertainty = []
    #     for video in np.unique(files):
    #         current_data = x[files == video]
    #         current_labels = y[files == video]
    #         true_label = current_labels[0]
    #         current_logits = model.predict(current_data)
    #         mc_logits = []
    #         for i in range(50):
    #             mc_logits.append(mc_model.predict(current_data))
    #         mc_uncertainty.append(np.max(np.std(mc_logits, axis=0)))
    #         patient_logits = np.mean(current_logits, axis=0)
    #         gt.append(true_label)
    #         logits.append(patient_logits)
    #         videos.append(video)
    #     logits = np.array(logits)
    #     print(logits)
    #     gt = np.array(gt)
    #     S = 3 + np.sum(logits, axis=1)
    #     u = 3 / S
    #     probs = (logits + 1) / S[:, None]
    #     plot_loss_vs_uncertainty(labels=gt, loss=np.sum(np.abs(gt - probs), axis=1), uncertainty=u,
    #                              start_of_filename="evidential")
    #     prediction_accuracies = np.argmax(gt, axis=1) == np.argmax(probs, axis=1)
    #     plt.figure()
    #     plot_rar_vs_rer(prediction_accuracies, u, tag='evidential', color='blue', m='*')
    #     plot_rar_vs_rer(prediction_accuracies, mc_uncertainty, tag='mc', color='red', m='o')
    #     print('these are the ones I wanna cutoff')
    #     for i in range(len(u)):
    #         if u[i] > 0.2:
    #             print(videos[i])

    # print("-----------------------------TRAINING-----------------------------")
    # train_gt, train_preds = calculate_patient_wise(train_files, X_train, Y_train, model)
    # print("-----------------------------VALIDATION-----------------------------")
    # validation_gt, validation_preds = calculate_patient_wise(validation_files, X_validation, Y_validation, model)
    # print("-----------------------------TESTING-----------------------------")
    # test_gt, test_preds = calculate_patient_wise(test_files, X_test, Y_test, model)

    # if args.uncertainty:
    #     print('-------------------------------uncertainty-----------------------------------')
    #     patient_wise_uncertainty(test_files, X_test, Y_test, model)

    # printAndSaveClassificationReport(train_gt, train_preds, lb.classes_, "trainReportPatients.csv")
    # printAndSaveClassificationReport(validation_gt, validation_preds, lb.classes_, "validationReportPatients.csv")
    # printAndSaveClassificationReport(test_gt, test_preds, lb.classes_, "testReportPatients.csv")
    # printAndSaveConfusionMatrix(train_gt, train_preds, lb.classes_, "trainConfusionMatrixPatients.png")
    # printAndSaveConfusionMatrix(validation_gt, validation_preds, lb.classes_, "validationConfusionMatrixPatients.png")
    # printAndSaveConfusionMatrix(test_gt, test_preds, lb.classes_, "testConfusionMatrixPatients.png")

    # plot the training loss and accuracy
    # plt.style.use('ggplot')
    # plt.figure()
    # plt.plot(np.arange(0, len(H.history['loss'])), H.history['loss'], label='train_loss')
    # plt.plot(np.arange(0, len(H.history['val_loss'])), H.history['val_loss'], label='val_loss')
    # plt.plot(np.arange(0, len(H.history['accuracy'])), H.history['accuracy'], label='train_acc')
    # plt.plot(np.arange(0, len(H.history['val_accuracy'])), H.history['val_accuracy'], label='val_acc')
    # plt.title('Training Loss and Accuracy on COVID-19 Dataset')
    # plt.xlabel('Epoch #')
    # plt.ylabel('Loss/Accuracy')
    # plt.legend(loc='lower left')
    # plt.savefig(os.path.join(FINAL_OUTPUT_DIR, 'loss.png'))
    print('-------------------------------Aggregated Results-----------------------------------')

    val_true_bline = np.concatenate(np.array(val_true[0]), axis=None)
    val_true_aline = np.concatenate(np.array(val_true[1]), axis=None)
    val_pred_bline = np.concatenate(np.array(val_pred[0]), axis=None)
    val_pred_aline = np.concatenate(np.array(val_pred[1]), axis=None)
    test_true_bline = np.concatenate(np.array(test_true[0]), axis=None)
    test_true_aline = np.concatenate(np.array(test_true[1]), axis=None)
    test_pred_bline = np.concatenate(np.array(test_pred[0]), axis=None)
    test_pred_aline = np.concatenate(np.array(test_pred[1]), axis=None)
    cm = confusion_matrix(val_true_aline, val_pred_aline)
    print("validation, aline")
    print(classification_report(val_true_aline, val_pred_aline))
    print(cm)

    cm = confusion_matrix(val_true_bline, val_pred_bline)
    print("validation, bline")
    print(classification_report(val_true_bline, val_pred_bline))
    print(cm)

    cm = confusion_matrix(test_true_aline, test_pred_aline)
    print("test, aline")
    print(classification_report(test_true_aline, test_pred_aline))
    print(cm)

    cm = confusion_matrix(test_true_bline, test_pred_bline)
    print("test, bline")
    print(classification_report(test_true_bline, test_pred_bline))
    print(cm)

    # printAndSaveClassificationReport(val_true, val_pred, lb.classes_, "val_CV.csv")
    # printAndSaveClassificationReport(test_true, test_pred, lb.classes_, "test_CV.csv")


if __name__ == '__main__':
    main()
