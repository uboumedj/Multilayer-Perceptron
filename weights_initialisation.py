import numpy as np
import pandas as pd
import math


def he_uniform(shape):
    """
    Initialises the weights of a layer using the He Uniform method
    Arguments:
        shape (tuple): the shape of the layer's input and output units
    Returns:
        A vector containing the generated weights
    """
    if (not isinstance(shape, tuple) or len(shape) != 2):
        print(f"error: wrong input shape while initialising weights")
        return None
    nb_of_input_units = shape[0]
    scale = 2.0 / max(1.0, nb_of_input_units)
    limit = math.sqrt(3.0 * scale)
    return np.random.uniform(low=-limit, high=limit, size=shape)


def he_normal(shape):
    """
    Initialises the weights of a layer using the He Normal method
    Arguments:
        shape (tuple): the shape of the layer's input and output units
    Returns:
        A vector containing the generated weights
    """
    if (not isinstance(shape, tuple) or len(shape) != 2):
        print(f"error: wrong input shape while initialising weights")
        return None
    nb_of_input_units = shape[0]
    scale = 2.0 / max(1.0, nb_of_input_units)
    deviation = math.sqrt(scale) / 0.87962566103423978
    return np.random.normal(loc=0.0, scale=deviation, size=shape)


def xavier_uniform(shape):
    """
    Initialises the weights of a layer using the Xavier(Glorot)
    Uniform method
    Arguments:
        shape (tuple): the shape of the layer's input and output units
    Returns:
        A vector containing the generated weights
    """
    if (not isinstance(shape, tuple) or len(shape) != 2):
        print(f"error: wrong input shape while initialising weights")
        return None
    nb_of_input_units = shape[0]
    nb_of_output_units = shape[1]
    scale = 1.0 / max(1.0, (nb_of_input_units + nb_of_output_units) / 2)
    limit = math.sqrt(3.0 * scale)
    return np.random.uniform(low=-limit, high=limit, size=shape)


def xavier_normal(shape):
    """
    Initialises the weights of a layer using the Xavier(Glorot)
    Normal method
    Arguments:
        shape (tuple): the shape of the layer's input and output units
    Returns:
        A vector containing the generated weights
    """
    if (not isinstance(shape, tuple) or len(shape) != 2):
        print(f"error: wrong input shape while initialising weights")
        return None
    nb_of_input_units = shape[0]
    nb_of_output_units = shape[1]
    scale = 1.0 / max(1.0, (nb_of_input_units + nb_of_output_units) / 2)
    deviation = math.sqrt(scale) / 0.87962566103423978
    return np.random.normal(loc=0.0, scale=deviation, size=shape)


def random_normal(shape, deviation=0.01):
    """
    Initialises the weights of a layer with random values spread
    in a Normal manner
    Arguments:
        shape (tuple): the shape of the layer's input and output units
        deviation (float): the expected spread of the values
    Returns:
        A vector containing the generated weights
    """
    if (not isinstance(shape, tuple) or len(shape) != 2):
        print(f"error: wrong input shape while initialising weights")
        return None
    if (not isinstance(deviation, float) or deviation < 0.0):
        print(f"error: weight init deviation must be a positive float")
        return None
    return np.random.normal(loc=0.0, scale=deviation, size=shape)


def random_uniform(shape, deviation=0.05):
    """
    Initialises the weights of a layer with random values spread
    in a Uniform manner
    Arguments:
        shape (tuple): the shape of the layer's input and output units
        deviation (float): the expected spread of the values
    Returns:
        A vector containing the generated weights
    """
    if (not isinstance(shape, tuple) or len(shape) != 2):
        print(f"error: wrong input shape while initialising weights")
        return None
    if (not isinstance(deviation, float) or deviation < 0.0):
        print(f"error: weight init deviation must be a positive float")
        return None
    return np.random.uniform(low=-deviation, high=deviation, size=shape)
