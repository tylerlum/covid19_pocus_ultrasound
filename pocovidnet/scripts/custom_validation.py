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


if __name__ == '__main__':
    # Inputs
    data_folder = '/home/tylergwlum/Aline-Bline (POCUS Dataset labeled by Tyler and Mobina)/convex'
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
    if task == 'a_lines':
        raw_data, raw_labels, raw_files = vid3d.video3d(video_files, a_lines, save=None)
    elif task == 'b_lines':
        raw_data, raw_labels, raw_files = vid3d.video3d(video_files, b_lines, save=None)

    # Preprocess X and Y
    X = preprocess_video_dataset(raw_data, 'vgg16')

    lb = LabelBinarizer()
    lb.fit(raw_labels)
    Y = np.array(lb.transform(raw_labels))
    if Y.shape[1] == 1:
        Y = tf.keras.utils.to_categorical(Y, num_classes=2, dtype=Y.dtype)

    print(f"X.shape = {X.shape}")
    print(f"Y.shape = {Y.shape}")

    # Load models
    aline_model_1 = 'video_model_outputs/Apr-12-2021_00-57-44/last_epoch'
    bline_model_1 = 'video_model_outputs/Apr-12-2021_00-59-34/last_epoch'
    if task == 'a_lines':
        selected_model = aline_model_1
    elif task == 'b_lines':
        selected_model = bline_model_1
    transferred_model = tf.keras.models.load_model(selected_model, custom_objects={'TransformerBlock': TransformerBlock})

    # Evaluate model
    transferred_model.evaluate(X, Y)
    predictions = transferred_model.predict(X)
    predIdxs = np.argmax(predictions, axis=1)
    trueIdxs = np.argmax(Y, axis=1)

    def printAndSaveConfusionMatrix(trueIdxs, predIdxs, classes, confusionMatrixFilename, directory=output_directory):
        print(f'confusion matrix for {confusionMatrixFilename}')
        cm = confusion_matrix(trueIdxs, predIdxs, labels=np.arange(len(classes)))
        # show the confusion matrix, accuracy, sensitivity, and specificity
        print(cm)

        plt.figure()
        cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        cmDisplay.plot()
        plt.savefig(os.path.join(directory, confusionMatrixFilename))

    def printAndSaveClassificationReport(trueIdxs, predIdxs, classes, reportFilename, directory=output_directory):
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

    if task == 'a_lines':
        classes = ['No Alines', 'Alines']
    elif task == 'b_lines':
        classes = ['No Blines', 'Blines']

    printAndSaveConfusionMatrix(trueIdxs, predIdxs, classes, 'confusion_matrix.png')
    printAndSaveClassificationReport(trueIdxs, predIdxs, classes, 'classification_report.png')
