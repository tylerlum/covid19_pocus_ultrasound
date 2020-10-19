import tensorflow as tf
import os
import cv2
import argparse
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Dropout
from keras.models import Model, Input
from keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import numpy as np

IMG_WIDTH, IMG_HEIGHT = 224, 224

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    '-d', '--data_dir', required=True, help='path to input dataset'
)
ap.add_argument('-m', '--model_file', required=True, type=str, default='models/')
ap.add_argument(
    '-f', '--fold', type=int, default='0', help='evaluate on this fold'
)
args = vars(ap.parse_args())

# Initialize hyperparameters
DATA_DIR = args['data_dir']
MODEL_FILE = args['model_file']
FOLD = args['fold']

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

# Setup model
model = tf.keras.models.load_model(MODEL_FILE)
mc_model = create_mc_model(model)

# initialize the data augmentation object
augmentation = ImageDataGenerator(
    rotation_range=10,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1
)

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


logits = model.predict(data)
def accuracy(logits, labels):
    correct = np.sum(np.argmax(labels, axis=1) == np.argmax(logits, axis=1))
    total = labels.shape[0]
    return correct / total
print(f"Accuracy of deterministic = {accuracy(logits, labels)}")

NUM_RUNS = 20
all_logits = np.zeros((NUM_RUNS, labels.shape[0], labels.shape[1]))
accuracies = []
for i in range(NUM_RUNS):
    logits = mc_model.predict(data)
    accuracies.append(accuracy(logits, labels))
    all_logits[i, :, :] = logits

average_logits = np.mean(all_logits, axis=0)
print(f"Mean accuracy of individual mc models = {sum(accuracies)/len(accuracies)}")
print(f"Combined accuracy of mc models = {accuracy(average_logits, labels)}")

NUM_RUNS = 20
all_logits = np.zeros((NUM_RUNS, labels.shape[0], labels.shape[1]))
accuracies = []
augmented_image_generator = augmentation.flow(data, labels, shuffle=False, batch_size=1)

print(f"data.shape[0] = {data.shape[0]}")
for i in range(NUM_RUNS):
    logits = model.predict(augmented_image_generator, steps=data.shape[0])
    accuracies.append(accuracy(logits, labels))
    all_logits[i, :, :] = logits

average_logits = np.mean(all_logits, axis=0)
print(f"Mean accuracy of individual tta models = {sum(accuracies)/len(accuracies)}")
print(f"Combined accuracy of tta models = {accuracy(average_logits, labels)}")
