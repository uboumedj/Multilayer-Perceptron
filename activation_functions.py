import numpy as np
import pandas as pd
import math


def sigmoid(x):
    """
    Computes the sigmoid of a vector.
    Args:
        x (numpy.ndarray): vector of shape (m, 1).
    Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.size == 0:
        return None
    result = np.divide(1., 1. + np.exp(np.negative(x)))
    return result


def sigmoid_gradient(x, next_layer_gradient, cached_result):
    """
    Computes the gradient of the Sigmoid 'part' of the layer
    for back-propagation
    Args:
        x (numpy.ndarray): vector of shape (m, 1).
        next_layer_gradient (np.ndarray): the gradient received
        from the next layer
        cached_result (np.ndarray): potential cached result of the
        previously calculated sigmoid(x), to save time
    Returns:
        The computed gradient
        None if x is an empty numpy.ndarray.
    """
    if cached_result is not None:
        sigmoid_derivative = cached_result * (1. - cached_result)
    else:
        sigmoid_derivative = sigmoid(x) * (1. - sigmoid(x))
    result = next_layer_gradient * sigmoid_derivative
    return result


def softmax(x):
    """
    Applies the softmax function to a vector.
    Args:
        x (np.ndarray): vector of shape (m, 1)
    Returns:
        The computed vector, where softmax was applied. All
        elements of the vector will sum up to 1.
        None if x is an empty array.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.size == 0 or len(x.shape) != 2:
        return None
    max = np.max(x, axis=1)
    max = max[:, np.newaxis]
    numerator = np.exp(x - max)
    denominator = np.sum(numerator, axis=1)
    denominator = denominator[:, np.newaxis]
    result = numerator / denominator
    return result


def relu(x):
    """
    Computes the result of ReLU applied on a vector.
    Args:
        x (numpy.ndarray): vector of shape (m, 1).
    Returns:
        The ReLU value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.size == 0:
        return None
    result = np.maximum(0.0, x)
    return result


def relu_gradient(x, next_layer_gradient):
    """
    Computes the gradient of the ReLU 'part' of the layer
    for back-propagation
    Args:
        x (numpy.ndarray): vector of shape (m, 1).
        next_layer_gradient (np.ndarray): the gradient received
        from the next layer
    Returns:
        The computed gradient
        None if x is an empty numpy.ndarray.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.size == 0:
        return None
    relu_derivative = x > 0
    result = relu_derivative * next_layer_gradient
    return result


def tanh(x):
    """
    Computes the result of the tanh function applied on a vector
    Args:
        x (numpy.ndarray): vector of shape (m, 1).
    Returns:
        The ReLU value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.size == 0:
        return None
    x_ = np.array(x, copy=True)
    numerator = np.exp(x_) - np.exp(np.negative(x_))
    denominator = np.exp(x_) + np.exp(np.negative(x_))
    result = np.divide(numerator, denominator)
    return result


def tanh_gradient(x, next_layer_gradient, cached_result):
    """
    Computes the gradient of the TanH 'part' of the layer
    for back-propagation
    Args:
        x (numpy.ndarray): vector of shape (m, 1).
        next_layer_gradient (np.ndarray): the gradient received
        from the next layer
        cached_result (np.ndarray): potential cached result of the
        previously calculated tanh(x), to save time
    Returns:
        The computed gradient
        None if x is an empty numpy.ndarray.
    """
    if cached_result is not None:
        tanh_derivative = 1. - np.square(cached_result)
    else:
        tanh_derivative = 1. - np.square(tanh(x))
    result = next_layer_gradient * tanh_derivative
    return result
