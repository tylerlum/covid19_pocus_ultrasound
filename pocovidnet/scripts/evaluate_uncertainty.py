import tensorflow as tf
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

    return data, labels


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


def plot_loss_vs_uncertainty(loss, uncertainty, start_of_filename=None):
    output_filename = "loss_vs_uncertainty.png"
    if start_of_filename is not None:
        output_filename = start_of_filename + "_" + output_filename
    plt.style.use('ggplot')
    plt.figure()
    plt.scatter(uncertainty, loss)
    plt.title('L1 Loss vs. Uncertainty')
    plt.xlabel('Uncertainty')
    plt.ylabel('L1 Loss')
    plt.savefig(os.path.join(FINAL_OUTPUT_DIR, output_filename))


def plot_rar_vs_rer(accuracies, uncertainty_in_prediction, start_of_filename=None):
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

    output_filename = "rar_vs_rer.png"
    if start_of_filename is not None:
        output_filename = start_of_filename + "_" + output_filename
    plt.style.use('ggplot')
    plt.figure()
    plt.scatter(rers, rars)
    plt.title('RAR vs. RER')
    plt.xlabel('Remaining Error Rate (RER)')
    plt.ylabel('Remaining Accuracy Rate (RAR)')
    plt.savefig(os.path.join(FINAL_OUTPUT_DIR, output_filename))

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
    data, labels = get_dataset()

    # Consider shrinking dataset for easier understanding
    # start_idx, end_idx = 10, 14
    # data, labels = np.array(data[start_idx:end_idx]), np.array(labels[start_idx:end_idx])

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
        average_logits = np.mean(all_logits, axis=0)
        std_dev_logits = np.std(all_logits, axis=0, ddof=1)
        indices_of_prediction = np.argmax(average_logits, axis=1)
        uncertainty_in_prediction = np.take_along_axis(std_dev_logits, np.expand_dims(indices_of_prediction, axis=1), axis=-1).squeeze(axis=-1)
        print(f"Mean accuracy of individual mc models = {sum(accuracies)/len(accuracies)}")
        print(f"Combined accuracy of mc models = {accuracy(average_logits, labels)}")
        print(f"Average uncertainty in mc model predictions = {np.sum(uncertainty_in_prediction)/uncertainty_in_prediction.shape[0]}")

        # Plot accuracy vs uncertainty
        correct_labels = np.take_along_axis(labels, np.expand_dims(indices_of_prediction, axis=1), axis=-1).squeeze(axis=-1)
        l1_loss_of_prediction = np.absolute(correct_labels - np.max(average_logits, axis=1))
        plot_loss_vs_uncertainty(l1_loss_of_prediction, uncertainty_in_prediction, start_of_filename="mc_dropout")
        certain_and_incorrects, certain_and_corrects, uncertains = [], [], []
        for i, (l1_loss, uncertainty) in enumerate(zip(l1_loss_of_prediction, uncertainty_in_prediction)):
            incorrect = l1_loss > 0.90
            correct = l1_loss < 0.05
            certain = uncertainty < 0.07
            uncertain = uncertainty > 0.20
            if certain and incorrect:
                print(f"On index {i} found certain and incorrect")
                certain_and_incorrects.append(i)
            if certain and correct:
                print(f"On index {i} found certain and correct")
                certain_and_corrects.append(i)
            if uncertain:
                print(f"On index {i} found uncertain")
                uncertains.append(i)
        print(len(certain_and_incorrects))
        print(len(certain_and_corrects))
        print(len(uncertains))
        for n, index in enumerate(certain_and_incorrects):
            print(f"n = {n} certain_and_incorrects")
            if n > 10:
                break
            plt.imshow(data[index])
            plt.savefig(os.path.join(FINAL_OUTPUT_DIR, f"mc-dropout__certain-incorrect__index-{index}__label-{labels[index]}__uncertainty-{uncertainty_in_prediction[index]}__l1-loss-{l1_loss_of_prediction[index]}.png"))
        for n, index in enumerate(certain_and_corrects):
            print(f"n = {n} certain_and_corrects")
            if n > 10:
                break
            plt.imshow(data[index])
            plt.savefig(os.path.join(FINAL_OUTPUT_DIR, f"mc-dropout__certain-correct__index-{index}__label-{labels[index]}__uncertainty-{uncertainty_in_prediction[index]}__l1-loss-{l1_loss_of_prediction[index]}.png"))
        for n, index in enumerate(uncertains):
            print(f"n = {n} uncertains")
            if n > 10:
                break
            plt.imshow(data[index])
            plt.savefig(os.path.join(FINAL_OUTPUT_DIR, f"mc-dropout__uncertain__index-{index}__label-{labels[index]}__uncertainty-{uncertainty_in_prediction[index]}__l1-loss-{l1_loss_of_prediction[index]}.png"))


        # Plot RAR vs RER
        prediction_accuracies = np.argmax(labels, axis=1) == np.argmax(average_logits, axis=1)
        plot_rar_vs_rer(prediction_accuracies, uncertainty_in_prediction, start_of_filename="mc_dropout")

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
        for i in range(NUM_TEST_TIME_AUGMENTATION_RUNS):
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
