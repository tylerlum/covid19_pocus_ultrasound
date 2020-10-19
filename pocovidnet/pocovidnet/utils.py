import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import resample, shuffle
from tensorflow.keras.callbacks import Callback
import random


def undersample(X, Y, randomState=0, printText=""):
    # Separate datapoints by label
    classes = np.unique(np.argmax(Y, axis=1))
    indicesByClass = [np.where(np.argmax(Y, axis=1) == cls) for cls in classes]
    datapointsList = [X[indices] for indices in indicesByClass]
    labelsList = [Y[indices] for indices in indicesByClass]

    # Resample each class with the same # of samples as the class with least samples
    minimumNumClass = min([labels.shape[0] for labels in labelsList])
    finalDatapointsList, finalLabelsList = [], []
    for datapoints, labels in zip(datapointsList, labelsList):
        undersampledX, undersampledY = resample(datapoints, labels, n_samples=minimumNumClass, replace=False, random_state=randomState)
        finalDatapointsList.append(undersampledX)
        finalLabelsList.append(undersampledY)
    finalUndersampledX, finalUndersampledY = np.concatenate(finalDatapointsList), np.concatenate(finalLabelsList)

    # Shuffle the array
    shuffledX, shuffledY = shuffle(finalUndersampledX, finalUndersampledY, random_state=randomState)

    print(f"Previously had {X.shape[0]} {printText} samples, but now undersampled to: {shuffledX.shape[0]}")
    return shuffledX, shuffledY


def oversample(X, Y, randomState=0, printText=""):
    # Separate datapoints by label
    classes = np.unique(np.argmax(Y, axis=1))
    indicesByClass = [np.where(np.argmax(Y, axis=1) == cls) for cls in classes]
    datapointsList = [X[indices] for indices in indicesByClass]
    labelsList = [Y[indices] for indices in indicesByClass]

    # Resample each class with the same # of samples as the class with most samples
    maximumNumClass = max([labels.shape[0] for labels in labelsList])
    finalDatapointsList, finalLabelsList = [], []
    for datapoints, labels in zip(datapointsList, labelsList):
        oversampledX, oversampledY = resample(datapoints, labels, n_samples=maximumNumClass, replace=True, random_state=randomState)
        finalDatapointsList.append(oversampledX)
        finalLabelsList.append(oversampledY)
    finalOversampledX, finalOversampledY = np.concatenate(finalDatapointsList), np.concatenate(finalLabelsList)

    # Shuffle the array
    shuffledX, shuffledY = shuffle(finalOversampledX, finalOversampledY, random_state=randomState)
    print(f"Previously had {X.shape[0]} {printText} samples, but now oversampled to: {shuffledX.shape[0]}")
    return shuffledX, shuffledY


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

