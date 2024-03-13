import numpy as np
import pandas as pd
import math
import layers
from utilities import convert_predictions_to_labels
from utilities import binary_cross_entropy


class Model():
    def __init__(self, layer_list, inputs=None):
        self.layer_list = self._check_validity_layers(layer_list)
        self.finished_training = False
        self.epochs_trained = 0
        self.name = None
        self.acc_log = []
        self.val_acc_log = []
        self.loss_log = []
        self.val_loss_log = []

    def feed_forward(self, input):
        """
        Performs a pass through the network's layers with the given
        data
        Arguments:
            input (np.ndarray): the input data
        Returns:
            The activations for each layer (the last one being
            the network's resulting prediction)
        """
        activations = []
        for layer in self.layer_list:
            activations.append(layer.forward(input))
            input = activations[-1]
        return activations
    
    def predict(self, x):
        """
        Computes the predictions of the network on a dataset
        Arguments:
            x (np.ndarray): the input data
        Returns:
            The model's predictions
        """
        predictions = self.feed_forward(x)[-1]
        return predictions

    def train_one_epoch(self, X, y):    
        # Feed-forward part: get the network layers' activations
        layer_activations = self.feed_forward(X)
        layer_inputs = [X] + layer_activations
        y_pred = layer_activations[-1]
        
        # Compute the loss and the initial (or final?) gradient
        last_gradient = (y_pred - y) / y_pred.shape[0]

        # Back-propagate the gradient through the layers
        for i in range(len(self.layer_list))[::-1]:
            layer = self.layer_list[i]
            last_gradient = layer.backward(layer_inputs[i], last_gradient)
    
    def _check_validity_layers(self, layer_list):
        """
        Checks the validity of the layer list received during
        the instance's initialisation
        Arguments:
            layer_list (list): list of the model's layers
        Returns:
            The same list if it is valid, None otherwise
        """
        if (not isinstance(layer_list, list)):
            print(f"error: layer list must be a list of layers")
            return None
        for layer in layer_list:
            if (not isinstance(layer, layers.DenseLayer)):
                print(f"error: layer {layer} in list was not a layer")
                return None
        return layer_list


def createNetwork(layer_list):
    """
    Creates a Model object composed of the given layers
    Arguments:
        layer_list (list): list of the model's layers
    Returns:
        The resulting Model object
    """
    network = Model(layer_list=layer_list)
    return network


def separate_batches(x, y, batch_size):
    """
    Separates the dataset into random batches according to the
    user-specified batch size
    Arguments:
        x (np.ndarray): training dataset
        y (np.ndarray): target variables
        batch_size (int): size of the function's batches
    Yields:
        Batches until whole dataset has been cycled through
    """
    if len(x) != len(y):
        print(f"error: lengths of data and target are incompatible")
        return None
    indices = np.random.permutation(len(x))
    for start_idx in range(0, len(x) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield x[excerpt], y[excerpt]


def fit(network, x_train, y_train, x_valid, y_valid, epochs, batch_size):
    for epoch in range(epochs):
        # Train the model on each mini-batch
        for x_batch, y_batch in separate_batches(x_train, y_train, batch_size=batch_size):
            network.train_one_epoch(x_batch, y_batch)
        network.epochs_trained += 1

        # Compute intermediary metrics on training set and validation set
        predictions_train = network.predict(x_train)
        labels_train = convert_predictions_to_labels(predictions_train)
        network.acc_log.append(np.mean(labels_train == y_train))
        network.loss_log.append(binary_cross_entropy(y_train, predictions_train))

        predictions_valid = network.predict(x_valid)
        labels_valid = convert_predictions_to_labels(predictions_valid)
        network.val_acc_log.append(np.mean(labels_valid == y_valid))
        network.val_loss_log.append(binary_cross_entropy(y_valid, predictions_valid))

        print(f"epoch {epoch}/{epochs} - ", end="")
        print(f"loss: {network.loss_log[-1]:.5f} - ", end="")
        print(f"val_loss: {network.val_loss_log[-1]:.5f} - ", end="")
        print(f"acc: {network.acc_log[-1]:.5f} - ", end="")
        print(f"val_acc: {network.val_acc_log[-1]:.5f}")
    network.finished_training = True