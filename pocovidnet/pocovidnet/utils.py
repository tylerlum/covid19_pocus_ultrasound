import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import resample, shuffle
from tensorflow.keras.callbacks import Callback
import random


def undersample(X, Y, randomState=0, printText="", perform_shuffle=True):
    # Separate datapoints by label
    classes = np.unique(np.argmax(Y, axis=1))
    indicesByClass = [np.where(np.argmax(Y, axis=1) == cls) for cls in classes]
    datapointsList = [X[indices] for indices in indicesByClass]
    labelsList = [Y[indices] for indices in indicesByClass]

    # Resample each class with the same # of samples as the class with most samples
    minimumNumClass = min([labels.shape[0] for labels in labelsList])
    finalUndersampledX = np.zeros((minimumNumClass*len(classes), *X.shape[1:]), dtype=X.dtype)
    finalUndersampledY = np.zeros((minimumNumClass*len(classes), *Y.shape[1:]), dtype=Y.dtype)
    currentIndex = 0
    for datapoints, labels in zip(datapointsList, labelsList):
        undersampledX, undersampledY = resample(datapoints, labels, n_samples=minimumNumClass, replace=False, random_state=randomState)
        nextCurrentIndex = currentIndex + minimumNumClass
        finalUndersampledX[currentIndex:nextCurrentIndex] = undersampledX
        finalUndersampledY[currentIndex:nextCurrentIndex] = undersampledY
        currentIndex = nextCurrentIndex

    # Shuffle the array
    if perform_shuffle:
        finalUndersampledX, finalUndersampledY = shuffle(finalUndersampledX, finalUndersampledY, random_state=randomState)

    print(f"Previously had {X.shape[0]} {printText} samples, but now undersampled to: {finalUndersampledX.shape[0]}")
    return finalUndersampledX, finalUndersampledY


def oversample(X, Y, randomState=0, printText="", perform_shuffle=True):
    # Separate datapoints by label
    classes = np.unique(np.argmax(Y, axis=1))
    indicesByClass = [np.where(np.argmax(Y, axis=1) == cls) for cls in classes]
    datapointsList = [X[indices] for indices in indicesByClass]
    labelsList = [Y[indices] for indices in indicesByClass]

    # Resample each class with the same # of samples as the class with most samples
    maximumNumClass = max([labels.shape[0] for labels in labelsList])
    finalOversampledX = np.zeros((maximumNumClass*len(classes), *X.shape[1:]), dtype=X.dtype)
    finalOversampledY = np.zeros((maximumNumClass*len(classes), *Y.shape[1:]), dtype=Y.dtype)
    currentIndex = 0
    for datapoints, labels in zip(datapointsList, labelsList):
        oversampledX, oversampledY = resample(datapoints, labels, n_samples=maximumNumClass, replace=True, random_state=randomState)
        nextCurrentIndex = currentIndex + maximumNumClass
        finalOversampledX[currentIndex:nextCurrentIndex] = oversampledX
        finalOversampledY[currentIndex:nextCurrentIndex] = oversampledY
        currentIndex = nextCurrentIndex

    # Shuffle the array
    if perform_shuffle:
        finalOversampledX, finalOversampledY = shuffle(finalOversampledX, finalOversampledY, random_state=randomState)

    print(f"Previously had {X.shape[0]} {printText} samples, but now oversampled to: {finalOversampledX.shape[0]}")
    return finalOversampledX, finalOversampledY


# A class to show balanced accuracy.
class Metrics(Callback):

    def __init__(self, valid_data, model):
        super(Metrics, self).__init__()
        self.valid_data = valid_data
        self._data = []
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        # if epoch:
        #     for i in range(1):  # len(self.valid_data)):
        x_test_batch, y_test_batch = self.valid_data

        y_predict = np.asarray(self.model.predict(x_test_batch))

        y_val = np.argmax(y_test_batch, axis=1)
        y_predict = np.argmax(y_predict, axis=1)
        self._data.append(
            {
                'val_balanced': balanced_accuracy_score(y_val, y_predict),
            }
        )
        print(f'Balanced accuracy is: {self._data[-1]}')
        return

    def get_data(self):
        return self._data


def fix_layers(model, num_flex_layers: int = 1):
    """
    Receives a model and freezes all layers but the last num_flex_layers ones.

    Arguments:
        model {tensorflow.python.keras.engine.training.Model} -- model

    Keyword Arguments:
        num_flex_layers {int} -- [Number of trainable layers] (default: {1})

    Returns:
        Model -- updated model
    """
    num_layers = len(model.layers)
    for ind, layer in enumerate(model.layers):
        if ind < num_layers - num_flex_layers:
            layer.trainable = False

    return model

