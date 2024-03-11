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


def sigmoid_gradient(x, next_layer_gradient, cached_sigmoid_result):
    if cached_sigmoid_result is not None:
        sigmoid_derivative = cached_sigmoid_result * (1. - cached_sigmoid_result)
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
    s = np.max(x, axis=1)
    s = s[:, np.newaxis]
    numerator = np.exp(x - s)
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
