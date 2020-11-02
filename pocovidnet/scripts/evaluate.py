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
    ap.add_argument('-s', '--shap', type=bool, default=False)
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
    SHAP = args['shap']
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

        # Create regular model
        model_path = os.path.join(MODEL_DIR, "model-0", MODEL_FILE)
        print(f"Looking for model at {model_path}")
        model = tf.keras.models.load_model(model_path)

        # make predictions on the testing set
        logits = model.predict(data)
        save_evaluation_files(labels, logits, classes, "regular", FINAL_OUTPUT_DIR)

    if SHAP:
        # Create regular model
        model_path = os.path.join(MODEL_DIR, "model-0", MODEL_FILE)
        print(f"Looking for model at {model_path}")
        model = tf.keras.models.load_model(model_path)

        # Selected images for evaluation
        folder = "outputs/scripts|evaluate_uncertainty.py__Data-..|data|cross_validation_balanced__Model-models|ensemble_balanced3|validation-fold-2_test-fold-3__Fold-3__MC-True__TTA-False__ENSEMBLE-False__Oct-27-2020_18-02-19"
        image_filenames = ["mc_dropout__certain-incorrect__index-84__label-[1 0 0]__uncertainty-0.0691902687061081__l1-loss-0.906338838338852.png", "mc_dropout__certain-incorrect__index-180__label-[1 0 0]__uncertainty-0.01399976156860364__l1-loss-0.9907586216926575.png", "mc_dropout__certain-incorrect__index-185__label-[0 1 0]__uncertainty-0.04448785803094582__l1-loss-0.9283356690406799.png", "mc_dropout__certain-incorrect__index-127__label-[0 0 1]__uncertainty-0.020174342382949104__l1-loss-0.9932182478904724.png", "mc_dropout__certain-incorrect__index-20__label-[0 0 1]__uncertainty-0.008452541030487913__l1-loss-0.9952394783496856.png", "mc_dropout__certain-incorrect__index-58__label-[0 0 1]__uncertainty-0.03691961227103745__l1-loss-0.9810117626190186.png", "mc_dropout__certain-correct__index-21__label-[1 0 0]__uncertainty-0.02249087042797793__l1-loss-0.01452756881713868.png", "mc_dropout__certain-correct__index-26__label-[1 0 0]__uncertainty-0.021063608159052588__l1-loss-0.009026633501052816.png", "mc_dropout__certain-correct__index-6__label-[0 1 0]__uncertainty-0.06496651852313849__l1-loss-0.026082087755203265.png", "mc_dropout__certain-correct__index-0__label-[0 1 0]__uncertainty-0.007189998127695973__l1-loss-0.005185798406600939.png", "mc_dropout__certain-correct__index-7__label-[0 0 1]__uncertainty-0.00873774073540707__l1-loss-0.0038542127609253463.png", "mc_dropout__certain-correct__index-5__label-[0 0 1]__uncertainty-0.015318109900374043__l1-loss-0.01686740398406983.png", "mc_dropout__most-uncertain__index-218__label-[1 0 0]__uncertainty-0.11493162844491756__l1-loss-0.6389305627346039.png", "mc_dropout__most-uncertain__index-149__label-[1 0 0]__uncertainty-0.141019137374305__l1-loss-0.6775126373767852.png", "mc_dropout__most-uncertain__index-190__label-[0 1 0]__uncertainty-0.15890519701658554__l1-loss-0.26360798537731167.png", "mc_dropout__most-uncertain__index-208__label-[0 1 0]__uncertainty-0.1501838342272395__l1-loss-0.7401564568281174.png", "mc_dropout__most-uncertain__index-229__label-[0 0 1]__uncertainty-0.16262208517033447__l1-loss-0.6901898381114006.png", "mc_dropout__most-uncertain__index-124__label-[0 0 1]__uncertainty-0.1256323441802403__l1-loss-0.24258242070674896.png"]
        test_images = [os.path.join(folder, x) for x in image_filenames]
        test_images = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB) for x in test_images]
        test_images = [np.array(cv2.resize(x, (224, 224)))/255.0 for x in test_images]

        import shap
        explainer = shap.GradientExplainer(model, data)
        num_examples = 6
        num_full_plots = len(test_images) // num_examples
        print(f"len(test_images) = {len(test_images)}")
        for i in range(num_full_plots):
            lo, hi = i*num_examples, (i+1)*num_examples
            plt.figure()
            shap_values = explainer.shap_values(np.array(test_images[lo:hi]))
            shap.image_plot([shap_values[j] for j in range(len(shap_values))], np.array(test_images[lo:hi]))
            plt.savefig(os.path.join(FINAL_OUTPUT_DIR, f"shap{i}.png"))
        num_remaining = len(test_images) % num_examples
        if num_remaining > 0:
            lo = num_full_plots * num_examples
            plt.figure()
            shap_values = explainer.shap_values(np.array(test_images[lo:]))
            shap.image_plot([shap_values[j] for j in range(len(shap_values))], np.array(test_images[lo:]))
            plt.savefig(os.path.join(FINAL_OUTPUT_DIR, f"shap{num_full_plots}.png"))
        test_predictions = model.predict(np.array(test_images))
        print(f"test_predictions = {test_predictions}")



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
        NUM_TEST_TIME_AUGMENTATION_RUNS = 50
        print(f"Running {NUM_TEST_TIME_AUGMENTATION_RUNS} runs of Test Time Augmentation")
        print("========================")

        # Setup model with augmentation output
        model_path = os.path.join(MODEL_DIR, "model-0", MODEL_FILE)
        print(f"Looking for model at {model_path}")
        model = tf.keras.models.load_model(model_path)
        augmentation = ImageDataGenerator(
        )
        augmentation2 = ImageDataGenerator(
            brightness_range=(1.0, 1.0)
        )
        augmentation3 = ImageDataGenerator(
            brightness_range=(0.0, 0.0)
        )
        augmented_image_generator = augmentation.flow(data, labels, shuffle=False, batch_size=1)
        augmented_image_generator2 = augmentation2.flow(data, labels, shuffle=False, batch_size=1)
        augmented_image_generator3 = augmentation3.flow(data, labels, shuffle=False, batch_size=1)
        img, img2, img3 = augmented_image_generator[0][0][0], augmented_image_generator2[0][0][0], augmented_image_generator3[0][0][0]
        img4 = img2 / 255
        print(img.shape)
        print(img2.shape)
        print(img3.shape)
        print(img4.shape)
        print(data[0].shape)
        print(img.max())
        print(img2.max())
        print(img3.max())
        print(img4.max())
        print(data[0].max())
        print(np.average(img))
        print(np.average(img2))
        print(np.average(img3))
        print(np.average(img4))
        print(np.average(data[0]))
        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, "img.png"), img)
        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, "img-255.png"), img * 255)
        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, "img2.png"), img2)
        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, "img3.png"), img3)
        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, "img4.png"), img4)
        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, "data.png"), data[0])
        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, "data-255.png"), data[0] * 255)
        print(doit)


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
