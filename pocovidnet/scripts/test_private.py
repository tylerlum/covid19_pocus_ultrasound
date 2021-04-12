import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pocovidnet.videoto3d import Videoto3D
from pocovidnet.video_dataset_preprocess import preprocess_video_dataset
from sklearn.preprocessing import LabelBinarizer
from pocovidnet.transformer import TransformerBlock
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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


if __name__ == '__main__':
    # Inputs
    data_folder = '/home/mobinam/Aline-Bline (POCUS Dataset labeled by Tyler and Mobina)/convex'
    task = 'b_lines'  # 'a_lines' or 'b_lines'
    output_directory = 'test_output'
    # Setup output directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Get video paths and labels
    video_files = [os.path.join(data_folder, v) for v in os.listdir(data_folder)]
    b_lines = [int(v[1]) for v in os.listdir(data_folder)]
    a_lines = [int(v[4]) for v in os.listdir(data_folder)]
    labels = [(b, a) for b, a in zip(b_lines, a_lines)]  # (b_lines, a_lines)
    # Convert videos to numpy arrays
    vid3d = Videoto3D(data_folder, depth=9, framerate=3)
    # Create dataset
    # if task == 'a_lines':
    raw_data, raw_labels_a_lines, raw_files = vid3d.video3d(video_files, a_lines, save=None)
    # elif task == 'b_lines':
    raw_data, raw_labels_b_lines, _ = vid3d.video3d(video_files, b_lines, save=None)
    # Preprocess X and Y
    X = preprocess_video_dataset(raw_data, 'vgg16')
    # lb = LabelBinarizer()
    # lb.fit(raw_labels)
    Y = np.array([{'head_0': b, 'head_1': a} for b, a in zip(raw_labels_b_lines, raw_labels_a_lines)])
    # if Y.shape[1] == 1:
    #     Y = tf.keras.utils.to_categorical(Y, num_classes=2, dtype=Y.dtype)
    print(f"X.shape = {X.shape}")
    print(f"Y.shape = {Y.shape}")
    # Load models
    predictions = []
    for i in range(1):
        model_adress = 'multihead_model_outputs/Apr-12-2021_18-35-09/multihead_best_fold_{}'.format(i)
        transferred_model = tf.keras.models.load_model(model_adress,
                                                       custom_objects={'TransformerBlock': TransformerBlock,
                                                       'loss': weighted_categorical_crossentropy})
        predictions.append(transferred_model.predict(X))
    predictions = np.array(predictions) # 5 , 2, 172, 2
    predictions = np.sum(predictions, axis=0) # 2, 172 , 2
    head_0_preds = predictions[0] # 172, 2 -> bline
    head_1_preds = predictions[1] # 172, 2 -> aline
    gt_0 = np.array([t["head_0"] for t in Y])
    gt_1 = np.array([t["head_1"] for t in Y])
    pred_idx_0 = np.argmax(head_0_preds, axis=1)
    pred_idx_1 = np.argmax(head_1_preds, axis=1)
    cm = confusion_matrix(gt_0, pred_idx_0)
    print('results for head 0, blines')
    print(cm)
    print(classification_report(gt_0, pred_idx_0))

    cm = confusion_matrix(gt_1, pred_idx_1)
    print('results for head 1, alines')
    print(cm)
    print(classification_report(gt_1, pred_idx_1))
    print('*******')

