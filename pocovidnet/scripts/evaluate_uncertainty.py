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

def create_mc_model(model):
  layers = [l for l in model.layers]
  x = layers[0].output
  for i in range(1, len(layers)):
      # Replace dropout layers with MC dropout layers
      if isinstance(layers[i], Dropout):
          x = Dropout(0.5)(x, training=True)
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
    vertical_flip=True,
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
print(f"logits = {logits}")
print(f"labels = {labels}")
