import numpy as np
import pandas as pd
import math
import layers


class Model():
    def __init__(self, layer_list, inputs=None):
        self.layer_list = self._check_validity_layers(layer_list)
        self.finished_training = False

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


def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    # Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- reference_answers + softmax) / logits.shape[0]


def train(network, X, y):    
    # Feed-forward part: get the network layers' activations
    layer_activations = network.feed_forward(X)
    layer_inputs = [X] + layer_activations
    y_pred = layer_activations[-1]
    
    # Compute the loss and the initial (or final?) gradient
    #loss_grad = y_pred - y
    last_gradient = grad_softmax_crossentropy_with_logits(y_pred, y)

    # Back-propagate the gradient through the layers
    for i in range(len(network.layer_list))[::-1]:
        layer = network.layer_list[i]
        last_gradient = layer.backward(layer_inputs[i], last_gradient)
    network.finished_training = True
    