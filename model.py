import numpy as np
import pandas as pd
import math
import layers


class Model():
    def __init__(self, layer_list, inputs=None):
        self.layer_list = self._check_validity_layers(layer_list)
        self.inputs = inputs
        self.loss_history = []
        self.val_loss_history = []

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


def feed_forward(network, input):
    
    activations = []
    input = input
    for layer in network.layer_list:
        activations.append(layer.forward(input))
        input = activations[-1]
    
    assert len(activations) == len(network.layer_list)
    return activations


def binary_cross_entropy(y, y_pred):
    """
    Calculates the value of binary cross entropy cost function.
    Arguments:
        y (numpy.array): the true labels
        y_pred (numpy.array): the predicted labels
    Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta
        None if any argument is not of the expected type.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_pred, np.ndarray):
        return None
    if (y.size == 0 or y_pred.size == 0):
        return None
    y = y.astype(float)
    eps = 1e-15
    J_value = y * np.log(y_pred + eps)
    J_value += (1 - y) * np.log((1 - y_pred) + eps)
    J_value = (- np.sum(J_value) / y.shape[0])
    return J_value

def softmax_crossentropy_with_logits(logits, reference_answers):
    # Compute crossentropy from logits[batch,n_classes] and ids of correct answers
    logits_for_answers = logits[np.arange(len(logits)), reference_answers.astype(int)]
    
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
    
    return xentropy

def grad_softmax_crossentropy_with_logits(logits,reference_answers):
    # Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
    #ones_for_answers = np.zeros_like(logits)
    #ones_for_answers[np.arange(len(logits)), reference_answers.astype(int)] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- reference_answers + softmax) / logits.shape[0]


def loss_(y, y_hat):
    """
    Calculates the value of loss function.
    Arguments:
        y (numpy.array): a vector.
        y_hat (numpy.array): a vector.
    Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta
        None if any argument is not of the expected type.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if (y.size == 0 or y_hat.size == 0):
        return None
    y = y.astype(float)
    eps = 1e-15
    J_value = y * np.log(y_hat + eps)
    J_value += (1 - y) * np.log((1 - y_hat) + eps)
    J_value = (- np.sum(J_value) / y.shape[0])
    return J_value

def predict(network, X):
    # Compute network predictions. Returning indices of largest Logit probability
    predictions = feed_forward(network, X)[-1]
    return predictions

def train(network, X, y):    
    # Get the layer activations
    layer_activations = feed_forward(network, X)
    layer_inputs = [X] + layer_activations
    y_pred = layer_activations[-1]
    
    # Compute the loss and the initial gradient
    losse = loss_(y, y_pred)
    #loss_grad = y_pred - y
    loss_grad = grad_softmax_crossentropy_with_logits(y_pred, y)

    for layer_index in range(len(network.layer_list))[::-1]:
        layer = network.layer_list[layer_index]
        loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)
    return losse
    