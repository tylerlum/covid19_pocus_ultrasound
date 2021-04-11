import wandb
import math
from wandb.keras import WandbCallback
import argparse
import os
from tqdm import tqdm
import random
import imgaug
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import class_weight as cw
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
from pocovidnet.read_mat import loadmat
from pocovidnet.video_dataset_preprocess import preprocess_video_dataset
from pocovidnet.optical_flow import get_optical_flow_data
from pocovidnet.video_grad_cam_attention import VideoGradCAMAttention
from pocovidnet.video_grad_cam import VideoGradCAM
from pocovidnet.transfer_model import transferred_model_transformer, transferred_model_average, two_transferred_models
from pocovidnet.evidential import evidential_loss
from pocovidnet.weighted_class_weights import weighted_categorical_crossentropy
from datetime import datetime
from datetime import date
from keras.layers import Dropout, Dense, TimeDistributed, Input, GlobalAveragePooling1D, Concatenate
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


def plot_loss_vs_uncertainty(labels, loss, uncertainty, color_by_class=True, start_of_filename=None):
    output_filename = "loss_vs_uncertainty.png"
    if start_of_filename is not None:
        output_filename = start_of_filename + "_" + output_filename
    # Calculate values by class
    covid_loss = [loss[i] for i in range(len(loss)) if labels[i][0] == 1]
    pneu_loss = [loss[i] for i in range(len(loss)) if labels[i][1] == 1]
    reg_loss = [loss[i] for i in range(len(loss)) if labels[i][2] == 1]
    covid_uncertainty = [uncertainty[i] for i in range(len(uncertainty)) if labels[i][0] == 1]
    pneu_uncertainty = [uncertainty[i] for i in range(len(uncertainty)) if labels[i][1] == 1]
    reg_uncertainty = [uncertainty[i] for i in range(len(uncertainty)) if labels[i][2] == 1]

    colors = ['red', 'yellow', 'blue']
    mylabels = ['covid', 'pneu', 'reg']
    losses = [covid_loss, pneu_loss, reg_loss]
    uncertainties = [covid_uncertainty, pneu_uncertainty, reg_uncertainty]
    if color_by_class:
        plt.style.use('ggplot')
        plt.figure()
        for i in range(len(colors)):
            # print(uncertainties[i])
            # print(losses[i])
            plt.scatter(uncertainties[i], losses[i], c=colors[i], label=mylabels[i])
        plt.title('L1 Loss vs. Uncertainty')
        plt.xlabel('Uncertainty')
        plt.ylabel('L1 Loss')
        plt.legend()
        plt.savefig(output_filename)
    else:
        plt.style.use('ggplot')
        plt.figure()
        plt.scatter(uncertainty, loss)
        plt.title('L1 Loss vs. Uncertainty')
        plt.xlabel('Uncertainty')
        plt.ylabel('L1 Loss')
        plt.savefig(output_filename)


def plot_rar_vs_rer(accuracies, uncertainty_in_prediction, tag, color, m):
    def get_rar_and_rer(certainties, accuracies):
        num_samples = accuracies.shape[0]

        num_certain_and_incorrect = sum(certainties * ~accuracies)
        num_certain_and_correct = sum(certainties * accuracies)

        return num_certain_and_correct/num_samples, num_certain_and_incorrect/num_samples

    rars, rers = [], []
    for uncertainty_threshold in np.arange(0, 1, 0.001):
        certainties = uncertainty_in_prediction < uncertainty_threshold
        rar, rer = get_rar_and_rer(certainties, accuracies)
        rars.append(rar)
        rers.append(rer)

    output_filename = "rar_vs_rer_{}.png".format(tag)
    plt.style.use('ggplot')
    # plt.figure()
    plt.scatter(rers, rars, color=color, marker=m, alpha=0.3)
    plt.title('RAR vs. RER')
    plt.xlabel('Remaining Error Rate (RER)')
    plt.ylabel('Remaining Accuracy Rate (RAR)')
    plt.savefig(output_filename)


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

    # Private dataset setup
    parser.add_argument('--mat', type=str2bool, nargs='?', const=True, default=False,
                        help=('for pocovidnet dataset, do not use. Only used for reading mat files,' +
                              'pass in the lung/labelled folder to --videos'))
    parser.add_argument('--multitask', type=str2bool, nargs='?', const=True, default=False,
                        help='perform prediction on a_lines and b_lines_binary tasks. If set, ignores mat_task.')
    parser.add_argument('--mat_task', type=str, default=False,
                        help='perform prediction on this task: a_lines, b_lines, b_lines_binary')

    # Transfer between tasks
    parser.add_argument("--transferred_model", type=str, default="",
                        help="path to knowledge transferred model, ignored if empty string")
    parser.add_argument("--transferred_model2", type=str, default="",
                        help="path to second knowledge transferred model, ignored if empty string")
    parser.add_argument("--transferred_model_num_layers_remove", type=int, default=0,
                        help="number of layers to remove from transferred model (excluding prediction head that will always be removed)")
    parser.add_argument("--transferred_model_num_layers_add", type=int, default=0,
                        help="number of Dense layers to add to transferred model (includes a dropout after each)")
    parser.add_argument("--remove_pneumonia", type=str2bool, nargs='?', const=True, default=False,
                        help="remove pneumonia class")

    # Explainability
    parser.add_argument("--explain", type=str2bool, nargs='?', const=True, default=False,
                        help="save explanation heatmaps for video classification. Requires setting transferred_model to a CNN_transformer model with a TimeDistributed layer output shape (batch, seq_len, height, width, channels) and an TransformerBlock output shape (batch, seq_len, embed_dim)")
    parser.add_argument("--skip_data", type=str2bool, nargs='?', const=True, default=False,
                        help="skip a bunch of data to save time")

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
    parser.add_argument('--test_fold', type=int, default=0,
                        help=('fold for test. set to -1 to run on all folds and average results.' +
                              'validation = (test_fold+1)%num_folds'))

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
    parser.add_argument('--trainable_base', type=str2bool, nargs='?', const=True, default=False,
                        help='set if base model is trainable')
    parser.add_argument('--time_aggregation', type=str, default="pooling",
                        help='pooling or conv1d, aggregation across time for transformer')

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

    if "transformer" not in args.architecture:
        print("This model does not use time_aggregation")
        args.time_aggregation = None
    else:
        print(f"This model uses time_aggregation {args.time_aggregation}")

    if args.multitask:
        print('this is a multitask network')
    else:
        print('this is a regular network, not a multitask one')

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

    # Setup folds
    test_folds = [args.test_fold] if args.test_fold != -1 else [t for t in range(args.num_folds)]
    print(f"Going to run on these test folds {test_folds}")
    trainPredIdxsList, validationPredIdxsList, testPredIdxsList = [], [], []
    trainTrueIdxsList, validationTrueIdxsList, testTrueIdxsList = [], [], []
    trainPatientPredIdxsList, validationPatientPredIdxsList, testPatientPredIdxsList = [], [], []
    trainPatientTrueIdxsList, validationPatientTrueIdxsList, testPatientTrueIdxsList = [], [], []
    for test_fold in test_folds:
        validation_fold = (test_fold + 1) % args.num_folds  # Select validation fold
        print()
        print("===========================")
        print(f"Performing k-fold splitting with validation fold {validation_fold} and test fold {test_fold}")
        print("===========================")
        k_fold_cross_validation = StratifiedKFold(n_splits=args.num_folds, random_state=args.random_seed, shuffle=True)
        # k_fold_cross_validation = KFold(n_splits=args.num_folds, random_state=args.random_seed, shuffle=True)

        # StratifiedKFold Doesn't work when not enough datapoints of each class
        if args.skip_data:
            k_fold_cross_validation = KFold(n_splits=args.num_folds, random_state=args.random_seed, shuffle=True)

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
            class_short = ["cov", "pne", "reg"]
            if args.remove_pneumonia:
                class_short = ["cov", "reg"]
            vid_files = [
                v for v in os.listdir(args.videos) if v[:3].lower() in class_short
            ]
            labels = [vid[:3].lower() for vid in vid_files]

            # Read in videos and transform to 3D
            print()
            print("===========================")
            print("Reading in videos")
            print("===========================")

            train_files, train_labels, validation_files, validation_labels, test_files, test_labels = (
                    get_train_validation_test_split(validation_fold, test_fold, k_fold_cross_validation,
                                                    vid_files, labels)
                    )
            vid3d = Videoto3D(args.videos, width=args.width, height=args.height, depth=args.depth,
                              framerate=args.frame_rate, grayscale=args.grayscale, optical_flow_type=args.optical_flow_type)
            raw_train_data, raw_train_labels, train_files = vid3d.video3d(
                train_files,
                train_labels,
                save=None
            )
            raw_validation_data, raw_validation_labels, validation_files = vid3d.video3d(
                validation_files,
                validation_labels,
                save=None
            )
            raw_test_data, raw_test_labels, test_files = vid3d.video3d(
                test_files,
                test_labels,
                save=None
            )

            # Preprocess dataset
            X_train = preprocess_video_dataset(raw_train_data, args.pretrained_cnn)
            X_validation = preprocess_video_dataset(raw_validation_data, args.pretrained_cnn)
            X_test = preprocess_video_dataset(raw_test_data, args.pretrained_cnn)

            # One-hot encoding
            lb = LabelBinarizer()
            lb.fit(np.concatenate([raw_train_labels, raw_validation_labels, raw_test_labels], axis=None))
            Y_train = np.array(lb.transform(raw_train_labels))
            Y_validation = np.array(lb.transform(raw_validation_labels))
            Y_test = np.array(lb.transform(raw_test_labels))

            # Handle edge case of binary labels, generalize to softmax
            if Y_train.shape[1] == 1:
                Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=2, dtype=Y_train.dtype)
                Y_validation = tf.keras.utils.to_categorical(Y_validation, num_classes=2, dtype=Y_validation.dtype)
                Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=2, dtype=Y_test.dtype)

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

            if args.skip_data:
                all_mat_files = all_mat_files[:len(all_mat_files)//5]  # only used to save time

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

                    if args.multitask:
                        b_lines = tf.keras.utils.to_categorical(int(b_lines > 0), num_classes=2)
                        a_lines = tf.keras.utils.to_categorical(a_lines, num_classes=2)
                        labels.append({'head_0': b_lines,
                                       'head_1': a_lines})
                    else:
                        if args.mat_task == 'a_lines':
                            labels.append(a_lines)
                        elif args.mat_task == 'b_lines_binary':
                            labels.append(1 if b_lines > 0 else 0)
                        elif args.mat_task == 'b_lines':
                            labels.append(b_lines)
                return labels

            all_labels = get_labels(all_mat_files)

            train_mats, train_labels, validation_mats, validation_labels, test_mats, test_labels = (
                    get_train_validation_test_split(validation_fold, test_fold, k_fold_cross_validation,
                                                    all_mat_files, all_labels)
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

                    start_frame = 0
                    stop_frame = cine.shape[2] - 1
                    num_video_frames = (stop_frame - start_frame + 1) // show_every
                    num_clips = num_video_frames // args.depth

                    # Video clips
                    for i in range(num_clips):
                        start, stop = start_frame + i*args.depth*show_every, start_frame + (i+1)*args.depth*show_every
                        clip_data = cine[:, :, start:stop:show_every]
                        video_clip = []

                        # Frames
                        for frame_i in range(clip_data.shape[-1]):
                            frame = cv2.resize(clip_data[:, :, frame_i], (args.width, args.height))
                            frame = frame[:, :, np.newaxis]
                            frame = cv2.merge([frame, frame, frame])
                            video_clip.append(frame)

                        video_clips.append(video_clip)

                        if args.multitask:
                            b_lines_one_hot = tf.keras.utils.to_categorical(int(b_lines > 0), num_classes=2)
                            a_lines_one_hot = tf.keras.utils.to_categorical(a_lines, num_classes=2)
                            labels.append({'head_0': b_lines_one_hot,
                                           'head_1': a_lines_one_hot})
                        else:
                            if args.mat_task == 'a_lines':
                                labels.append(a_lines)
                            elif args.mat_task == 'b_lines_binary':
                                labels.append(1 if b_lines > 0 else 0)
                            elif args.mat_task == 'b_lines':
                                labels.append(b_lines)

                X = np.array(video_clips)

                # Bring together optical flow if needed
                if args.optical_flow_type is not None:
                    optical_flow_data = get_optical_flow_data(video_clips, args.optical_flow_type)
                    X = np.concatenate([X, optical_flow_data], axis=4)

                Y = np.array(labels)
                return X, Y

            # Get video clips and labels
            raw_train_data, raw_train_labels = get_video_clips_and_labels(train_mats)
            raw_validation_data, raw_validation_labels = get_video_clips_and_labels(validation_mats)
            raw_test_data, raw_test_labels = get_video_clips_and_labels(test_mats)

            # Preprocess dataset
            X_train = preprocess_video_dataset(raw_train_data, args.pretrained_cnn)
            X_validation = preprocess_video_dataset(raw_validation_data, args.pretrained_cnn)
            X_test = preprocess_video_dataset(raw_test_data, args.pretrained_cnn)

            # Onehot encode labels
            lb = LabelBinarizer()
            if args.multitask:
                Y_train = np.array(raw_train_labels)
                Y_validation = np.array(raw_validation_labels)
                Y_test = np.array(raw_test_labels)
            else:
                lb.fit(np.concatenate([raw_train_labels, raw_validation_labels, raw_test_labels], axis=None))
                Y_train = np.array(lb.transform(raw_train_labels))
                Y_validation = np.array(lb.transform(raw_validation_labels))
                Y_test = np.array(lb.transform(raw_test_labels))

            # Handle edge case of binary labels, generalize to softmax
            if not args.multitask and Y_train.shape[1] == 1:
                Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=2, dtype=Y_train.dtype)
                Y_validation = tf.keras.utils.to_categorical(Y_validation, num_classes=2, dtype=Y_validation.dtype)
                Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=2, dtype=Y_test.dtype)

            # Workaround to get text class names instead of numbers
            if args.multitask:
                lb.classes_ = ['B-lines', 'A-lines']
            else:
                if args.mat_task == 'a_lines':
                    lb.classes_ = ['No A-lines', 'A-lines']
                elif args.mat_task == 'b_lines_binary':
                    lb.classes_ = ['No B-lines', 'B-lines']
                elif args.mat_task == 'b_lines':
                    lb.classes_ = ['No B-lines', '1-2 B-lines', '3+ B-lines', 'Confluent B-lines']

        input_shape = X_train.shape[1:]
        print(f"input_shape = {input_shape}")

        # VISUALIZE
        if args.visualize:
            num_show = 1000
            print(f"Visualizing {num_show} video clips, storing them in {FINAL_OUTPUT_DIR}")
            for i in range(raw_train_data.shape[0]):
                # End early
                if i >= num_show:
                    break

                video_clip = raw_train_data[i]
                label = Y_train[i]
                for j in range(video_clip.shape[0]):
                    frame = video_clip[j]
                    num_channels = frame.shape[2]
                    if num_channels == 1 or num_channels == 3:
                        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Fold-{test_fold}_Example-{i}_Frame-{j}_Label-{label}.jpg"), frame)
                    elif num_channels == 6:
                        rgb_frame = frame[:, :, :3]
                        optical_flow_frame = frame[:, :, 3:]
                        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Fold-{test_fold}_Example-{i}_Frame-{j}_Label-{label}.jpg"),
                                    rgb_frame)
                        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Fold-{test_fold}_Example-{i}_Frame-{j}_Label-{label}-opt.jpg"),
                                    optical_flow_frame)

        # Need to pass in raw training data to generator so that it can perform augmentation on NOT preprocessed images,
        # then apply preprocessing after
        if args.augment:
            generator = DataGenerator(raw_train_data, Y_train, args.batch_size, input_shape, shuffle=True,
                                      pretrained_cnn=args.pretrained_cnn)

            if args.visualize:
                num_batches = 100
                print(f"Visualizing {num_batches} batch of augmented video clips")
                for k in range(num_batches):
                    batchX, batchY = generator[k]
                    for i, (video_clip, label) in enumerate(zip(batchX, batchY)):
                        for j, frame in enumerate(video_clip):
                            num_channels = frame.shape[2]
                            if num_channels == 1 or num_channels == 3:
                                cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Augment_Fold-{test_fold}_Example-{k*batchX.shape[0]+i}_Frame-{j}_Label-{label}.jpg"),
                                            frame)
                            elif num_channels == 6:
                                rgb_frame = frame[:, :, :3]
                                optical_flow_frame = frame[:, :, 3:]
                                cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Augment_Fold-{test_fold}_Example-{k*batchX.shape[0]+i}_Frame-{j}_Label-{label}.jpg"),
                                            rgb_frame)
                                cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f"Augment_Fold-{test_fold}_Example-{k*batchX.shape[0]+i}_Frame-{j}_Label-{label}-opt.jpg"),
                                            optical_flow_frame)

        print()
        print("===========================")
        print("Printing details about dataset")
        print("===========================")
        print(f"X_train.shape, Y_train.shape = {X_train.shape}, {Y_train.shape}")
        print(f"X_validation.shape, Y_validation.shape = {X_validation.shape}, {Y_validation.shape}")
        print(f"X_test.shape, Y_test.shape = {X_test.shape}, {Y_test.shape}")
        if not args.multitask:
            uniques, counts = np.unique(np.concatenate([raw_train_labels, raw_validation_labels, raw_test_labels], axis=None), return_counts=True)
            nb_classes = len(uniques)
            print(f"nb_classes, np.mean(X_train), np.mean(np.std(X_train, axis=(1,2,3,4))), np.max(X_train) = {nb_classes}, {np.mean(X_train)}, {np.mean(np.std(X_train, axis=(1,2,3,4)))}, {np.max(X_train)}")
            train_uniques, train_counts = np.unique(raw_train_labels, return_counts=True)
            validation_uniques, validation_counts = np.unique(raw_validation_labels, return_counts=True)
            test_uniques, test_counts = np.unique(raw_test_labels, return_counts=True)
            print("unique labels in train", (train_uniques, train_counts))
            print("unique labels in validation", (validation_uniques, validation_counts))
            print("unique labels in test", (test_uniques, test_counts))
            class_weight = dict(zip(np.arange(len(uniques)), cw.compute_class_weight('balanced', uniques, raw_train_labels)))
            print(f"class_weight = {class_weight}")
        else:
            nb_classes = len(lb.classes_)
            print(f'nb_classes = {nb_classes}')

        # Delete raw data we will not use (save RAM)
        del raw_train_data, raw_train_labels
        del raw_validation_data, raw_validation_labels
        del raw_test_data, raw_test_labels

        # Use transferred models
        if len(args.transferred_model) > 0:
            if len(args.transferred_model2) > 0:
                model = two_transferred_models(args.transferred_model, args.transferred_model2, args.architecture,
                                               input_shape, args.transferred_model_num_layers_add, args.trainable_base,
                                               nb_classes)
            elif "transformer" in args.architecture:
                model = transferred_model_transformer(args.transferred_model, args.transferred_model_num_layers_remove,
                                                      args.transferred_model_num_layers_add, args.trainable_base,
                                                      nb_classes)
            else:
                model = transferred_model_average(args.transferred_model, args.transferred_model_num_layers_remove,
                                                  args.transferred_model_num_layers_add, args.trainable_base,
                                                  args.nb_classes)
        # Not use transferred models
        else:
            if args.architecture in ["2D_CNN_average", "CNN_transformer"]:
                model = VIDEO_MODEL_FACTORY[args.architecture](input_shape, nb_classes, args.pretrained_cnn, args.time_aggregation, args.trainable_base)
            else:
                model = VIDEO_MODEL_FACTORY[args.architecture](input_shape, nb_classes, args.pretrained_cnn)

        # Use explainer
        if args.explain:
            print("TESTING GRAD CAMS AND ATTENTION EXPLAINER")
            from pocovidnet.transformer import TransformerBlock
            transferred_model = tf.keras.models.load_model(args.transferred_model, custom_objects={'TransformerBlock': TransformerBlock})
            explainer = VideoGradCAMAttention(transferred_model)
            cam = VideoGradCAM(transferred_model)
            # Run explainer on X examples
            for example in tqdm(range(len(X_train))):
                video = X_train[example]

                # New explainer method
                (heatmaps, overlays, attn_weights) = explainer.compute_attention_maps(video)

                # Old explainer method
                old_heatmaps = cam.compute_heatmaps(video)
                (old_heatmaps, old_overlays) = cam.overlay_heatmaps(old_heatmaps, video)

                # Save heatmaps overlayed on images
                images = []
                old_images = []
                for i in range(len(overlays)):
                    cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f'ex-{example}-overlays-{i}.jpg'), overlays[i])
                    images.append(cv2.imread(os.path.join(FINAL_OUTPUT_DIR, f'ex-{example}-overlays-{i}.jpg')))

                    cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f'ex-{example}-old_overlays-{i}.jpg'), old_overlays[i])
                    old_images.append(cv2.imread(os.path.join(FINAL_OUTPUT_DIR, f'ex-{example}-old_overlays-{i}.jpg')))
                plt.figure()
                plt.imshow(attn_weights)
                plt.savefig(os.path.join(FINAL_OUTPUT_DIR, f"ex-{example}-attn-weights.jpg"))

                # Save videos
                slower_frame_rate = 1
                output_video = cv2.VideoWriter(os.path.join(FINAL_OUTPUT_DIR, f'video-{example}.avi'), 0, slower_frame_rate, (args.width, args.height))
                old_output_video = cv2.VideoWriter(os.path.join(FINAL_OUTPUT_DIR, f'old-video-{example}.avi'), 0, slower_frame_rate, (args.width, args.height))
                for i in range(len(images)):
                    output_video.write(images[i])
                    old_output_video.write(old_images[i])
                output_video.release()
                old_output_video.release()

        print('---------------------------model---------------------\n', args.architecture)
        tf.keras.utils.plot_model(model, os.path.join(FINAL_OUTPUT_DIR, f"{args.architecture}.png"), show_shapes=True)

        if evidential:
            loss = evidential_loss(nb_classes)
        else:
            loss = categorical_crossentropy

        if args.optimizer == "adam":
            opt = Adam(lr=args.learning_rate)
        else:
            print(f"WARNING: invalid optimizer {args.optimizer}")

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


        model.compile(
            optimizer=opt, loss=loss, metrics=['accuracy']
        )

        wandb.init(entity='tylerlum', project=args.wandb_project)
        wandb.config.update(args)
        wandb.config.final_output_dir = FINAL_OUTPUT_DIR

        callbacks = [WandbCallback(save_model=args.save_model)]
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
            callbacks.append(ConfusionMatrixEachEpochCallback(X_validation, Y_validation, lb.classes_, fold=test_fold))

        print()
        print("===========================")
        print("About to train model")
        print("===========================")
        print(model.summary())

        if not args.multitask:
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
        else:
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
        # trainLoss, trainAcc = model.evaluate(X_train, Y_train, verbose=1)
        # print('train loss:', trainLoss)
        # print('train accuracy:', trainAcc)
        validationLoss, validationAcc = model.evaluate(X_validation, Y_validation, batch_size=args.batch_size, verbose=1)
        print('Validation loss:', validationLoss)
        print('Validation accuracy:', validationAcc)
        testLoss, testAcc = model.evaluate(X_test, Y_test, batch_size=args.batch_size, verbose=1)
        print('Test loss:', testLoss)
        print('Test accuracy:', testAcc)

        # rawTrainPredIdxs = model.predict(X_train, batch_size=args.batch_size, verbose=1)
        rawValidationPredIdxs = model.predict(X_validation, batch_size=args.batch_size, verbose=1)
        rawTestPredIdxs = model.predict(X_test, batch_size=args.batch_size, verbose=1)

        # for each image in the testing set we need to find the index of the
        # label with corresponding largest predicted probability
        # trainPredIdxs = np.argmax(rawTrainPredIdxs, axis=1)
        validationPredIdxs = np.argmax(rawValidationPredIdxs, axis=1)
        testPredIdxs = np.argmax(rawTestPredIdxs, axis=1)

        trainTrueIdxs = np.argmax(Y_train, axis=1)
        validationTrueIdxs = np.argmax(Y_validation, axis=1)
        testTrueIdxs = np.argmax(Y_test, axis=1)

        # Only do this with one fold, else they will overwrite eachother
        if len(test_folds) == 1:
            classes_with_validation = [f"{c} Validation" for c in lb.classes_]
            wandb.sklearn.plot_classifier(model, X_train, X_validation, trainTrueIdxs, validationTrueIdxs, validationPredIdxs,
                                          rawValidationPredIdxs, classes_with_validation, model_name=f"{args.architecture}")
            classes_with_test = [f"{c} Test" for c in lb.classes_]
            wandb.log({f'Test Confusion Matrix {test_fold}': wandb.plots.HeatMap(classes_with_test, classes_with_test,
                                                                                 matrix_values=confusion_matrix(testTrueIdxs, testPredIdxs, np.arange(len(lb.classes_))),
                                                                                 show_text=True)})

        # compute the confusion matrix and and use it to derive the raw
        # accuracy, sensitivity, and specificity
        def printAndSaveClassificationReport(trueIdxs, predIdxs, classes, reportFilename, directory=FINAL_OUTPUT_DIR, wandb_log=False):
            print(f'classification report sklearn for {reportFilename}')
            print(
                classification_report(
                    trueIdxs, predIdxs, target_names=classes, labels=np.arange(len(classes)),
                )
            )

            report = classification_report(
                trueIdxs, predIdxs, target_names=classes, labels=np.arange(len(classes)), output_dict=True
            )
            reportDf = pd.DataFrame(report).transpose()
            reportDf.to_csv(os.path.join(directory, reportFilename))

            if wandb_log:
                wandb_log_classification_table_and_plots(report, reportFilename)

        # printAndSaveClassificationReport(trainTrueIdxs, trainPredIdxs, lb.classes_, f"trainReport-{test_fold}.csv")
        printAndSaveClassificationReport(validationTrueIdxs, validationPredIdxs, lb.classes_, f"validationReport-{test_fold}.csv")
        printAndSaveClassificationReport(testTrueIdxs, testPredIdxs, lb.classes_, f"testReport-{test_fold}.csv")

        def printAndSaveConfusionMatrix(trueIdxs, predIdxs, classes, confusionMatrixFilename, directory=FINAL_OUTPUT_DIR, wandb_log=False):
            print(f'confusion matrix for {confusionMatrixFilename}')
            cm = confusion_matrix(trueIdxs, predIdxs, labels=np.arange(len(classes)))
            # show the confusion matrix, accuracy, sensitivity, and specificity
            print(cm)

            plt.figure()
            cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            cmDisplay.plot()
            plt.savefig(os.path.join(directory, confusionMatrixFilename))
            if wandb_log:
                wandb.log({confusionMatrixFilename: plt})

        # printAndSaveConfusionMatrix(trainTrueIdxs, trainPredIdxs, lb.classes_, f"trainConfusionMatrix-{test_fold}.png")
        printAndSaveConfusionMatrix(validationTrueIdxs, validationPredIdxs, lb.classes_, f"validationConfusionMatrix-{test_fold}.png")
        printAndSaveConfusionMatrix(testTrueIdxs, testPredIdxs, lb.classes_, f"testConfusionMatrix-{test_fold}.png")

        if args.save_model:
            print(f'Saving COVID-19 detector model on {FINAL_OUTPUT_DIR} data...')
            model_save_path = os.path.join(FINAL_OUTPUT_DIR, 'last_epoch')
            model.save(model_save_path)

            # Convert to tensorflow lite model
            print("About to make converter")
            converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)
            print("About to convert")
            tflite_model = converter.convert()
            tflite_model_path = "converted_model.tflite"
            print(f"About to write to file {tflite_model_path}")
            open(tflite_model_path, "wb").write(tflite_model)
            print("DONE")

        def calculate_patient_wise(files, x, y, model, verbose=True):
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
                if verbose:
                    print(f"video = {video}, true_label = {true_label}, prediction = {prediction}")
            return np.array(gt), np.array(preds)

        def create_mc_model(model, dropProb=0.5):
            layers = [layer for layer in model.layers]
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

        def patient_wise_uncertainty(files, x, y, model):
            gt = []
            logits = []
            videos = []
            files = np.array(files)
            mc_model = create_mc_model(model)
            mc_uncertainty = []
            for video in np.unique(files):
                current_data = x[files == video]
                current_labels = y[files == video]
                true_label = current_labels[0]
                current_logits = model.predict(current_data)
                mc_logits = []
                for i in range(50):
                    mc_logits.append(mc_model.predict(current_data))
                mc_uncertainty.append(np.max(np.std(mc_logits, axis=0)))
                patient_logits = np.mean(current_logits, axis=0)
                gt.append(true_label)
                logits.append(patient_logits)
                videos.append(video)
            logits = np.array(logits)
            print(logits)
            gt = np.array(gt)
            S = 3 + np.sum(logits, axis=1)
            u = 3/S
            probs = (logits+1)/S[:, None]
            plot_loss_vs_uncertainty(labels=gt, loss=np.sum(np.abs(gt - probs), axis=1), uncertainty=u,
                                     start_of_filename="evidential")
            prediction_accuracies = np.argmax(gt, axis=1) == np.argmax(probs, axis=1)
            plt.figure()
            plot_rar_vs_rer(prediction_accuracies, u, tag='evidential', color='blue', m='*')
            plot_rar_vs_rer(prediction_accuracies, mc_uncertainty, tag='mc', color='red', m='o')
            print('these are the ones I wanna cutoff')
            for i in range(len(u)):
                if u[i] > 0.2:
                    print(videos[i])

        if args.uncertainty:
            print('-------------------------------uncertainty-----------------------------------')
            patient_wise_uncertainty(test_files, X_test, Y_test, model)

        # Only works with public dataset
        if not args.mat:
            print("-----------------------------TRAINING-----------------------------")
            train_gt, train_preds = calculate_patient_wise(train_files, X_train, Y_train, model)
            print("-----------------------------VALIDATION-----------------------------")
            validation_gt, validation_preds = calculate_patient_wise(validation_files, X_validation, Y_validation, model)
            print("-----------------------------TESTING-----------------------------")
            test_gt, test_preds = calculate_patient_wise(test_files, X_test, Y_test, model)

            printAndSaveClassificationReport(train_gt, train_preds, lb.classes_, f"trainReportPatients-{test_fold}.csv")
            printAndSaveClassificationReport(validation_gt, validation_preds, lb.classes_, f"validationReportPatients-{test_fold}.csv")
            printAndSaveClassificationReport(test_gt, test_preds, lb.classes_, f"testReportPatients-{test_fold}.csv")
            printAndSaveConfusionMatrix(train_gt, train_preds, lb.classes_, f"trainConfusionMatrixPatients-{test_fold}.png")
            printAndSaveConfusionMatrix(validation_gt, validation_preds, lb.classes_, f"validationConfusionMatrixPatients-{test_fold}.png")
            printAndSaveConfusionMatrix(test_gt, test_preds, lb.classes_, f"testConfusionMatrixPatients-{test_fold}.png")

            trainPatientPredIdxsList.append(train_preds)
            validationPatientPredIdxsList.append(validation_preds)
            testPatientPredIdxsList.append(test_preds)
            trainPatientTrueIdxsList.append(train_gt)
            validationPatientTrueIdxsList.append(validation_gt)
            testPatientTrueIdxsList.append(test_gt)

        # Store results to aggregate
        # trainPredIdxsList.append(trainPredIdxs)
        validationPredIdxsList.append(validationPredIdxs)
        testPredIdxsList.append(testPredIdxs)
        # trainTrueIdxsList.append(trainTrueIdxs)
        validationTrueIdxsList.append(validationTrueIdxs)
        testTrueIdxsList.append(testTrueIdxs)

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
        plt.savefig(os.path.join(FINAL_OUTPUT_DIR, f'loss_fold-{test_fold}.png'))
        plt.style.use('default')

    # Aggregate results
    if len(test_folds) > 1:
        print('-------------------------------Aggregated Results-----------------------------------')
        # trainTrueIdxs = np.concatenate(trainTrueIdxsList, axis=None)
        # trainPredIdxs = np.concatenate(trainPredIdxsList, axis=None)
        validationTrueIdxs = np.concatenate(validationTrueIdxsList, axis=None)
        validationPredIdxs = np.concatenate(validationPredIdxsList, axis=None)
        testTrueIdxs = np.concatenate(testTrueIdxsList, axis=None)
        testPredIdxs = np.concatenate(testPredIdxsList, axis=None)
        # printAndSaveClassificationReport(trainTrueIdxs, trainPredIdxs, lb.classes_, "allTrainReport.csv", wandb_log=True)
        printAndSaveClassificationReport(validationTrueIdxs, validationPredIdxs, lb.classes_, "allValidationReport.csv", wandb_log=True)
        printAndSaveClassificationReport(testTrueIdxs, testPredIdxs, lb.classes_, "allTestReport.csv", wandb_log=True)
        # printAndSaveConfusionMatrix(trainTrueIdxs, trainPredIdxs, lb.classes_, "allTrainConfusionMatrix.png", wandb_log=True)
        printAndSaveConfusionMatrix(validationTrueIdxs, validationPredIdxs, lb.classes_, "allValidationConfusionMatrix.png", wandb_log=True)
        printAndSaveConfusionMatrix(testTrueIdxs, testPredIdxs, lb.classes_, "allTestConfusionMatrix.png", wandb_log=True)

        if not args.mat:
            # trainTrueIdxsPatients = np.concatenate(trainPatientTrueIdxsList, axis=None)
            # trainPredIdxsPatients = np.concatenate(trainPatientPredIdxsList, axis=None)
            validationTrueIdxsPatients = np.concatenate(validationPatientTrueIdxsList, axis=None)
            validationPredIdxsPatients = np.concatenate(validationPatientPredIdxsList, axis=None)
            testTrueIdxsPatients = np.concatenate(testPatientTrueIdxsList, axis=None)
            testPredIdxsPatients = np.concatenate(testPatientPredIdxsList, axis=None)
            # printAndSaveClassificationReport(trainTrueIdxsPatients, trainPredIdxsPatients, lb.classes_, "allTrainReportPatients.csv", wandb_log=True)
            printAndSaveClassificationReport(validationTrueIdxsPatients, validationPredIdxsPatients, lb.classes_, "allValidationReportPatients.csv", wandb_log=True)
            printAndSaveClassificationReport(testTrueIdxsPatients, testPredIdxsPatients, lb.classes_, "allTestReportPatients.csv", wandb_log=True)
            # printAndSaveConfusionMatrix(trainTrueIdxsPatients, trainPredIdxsPatients, lb.classes_, "allTrainConfusionMatrixPatients.png", wandb_log=True)
            printAndSaveConfusionMatrix(validationTrueIdxsPatients, validationPredIdxsPatients, lb.classes_, "allValidationConfusionMatrixPatients.png", wandb_log=True)
            printAndSaveConfusionMatrix(testTrueIdxsPatients, testPredIdxsPatients, lb.classes_, "allTestConfusionMatrixPatients.png", wandb_log=True)

        classes_with_test = [f"{c} Test" for c in lb.classes_]
        wandb.log({'Test Confusion Matrix': wandb.plots.HeatMap(classes_with_test, classes_with_test,
                                                                matrix_values=confusion_matrix(testTrueIdxs, testPredIdxs, np.arange(len(lb.classes_))),
                                                                show_text=True)})


if __name__ == '__main__':
    main()
