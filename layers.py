import numpy as np
import pandas as pd
import math
import weights_initialisation as wi
import activation_functions as af
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


class DenseLayer:
    def __init__(self, input_size, output_size, learning_rate=0.0314,
                 weights_initializer='heUniform', activation='sigmoid'):
        if (not isinstance(learning_rate, float) or learning_rate <= 0.0):
            print(f"warning: Layer learning rate <{learning_rate}> invalid. " +
                  "Reverting to default learning rate value of 0.0314")
            learning_rate = 0.0314
        if (not isinstance(input_size, int) or input_size <= 0):
            print(f"warning: Layer input size <{input_size}> invalid. " +
                  "Attempting input size value of 24 but it might not work")
            input_size = 24
        if (not isinstance(output_size, int) or output_size <= 0):
            print(f"warning: Layer output size <{output_size}> invalid. " +
                  "Attempting output size value of 24 but it might not work")
            output_size = 24
        self.learning_rate = learning_rate
        self.weights_initializer = weights_initializer
        self.weights = self._generate_weights(input_size, output_size)
        self.activation = activation
        self.cache = None
        if self.activation is not None:
            self.activation = activation.lower()
        self.biases = np.zeros(output_size)

    def forward(self, input):
        """
        Returns the layer's output for the feed-forward process. First
        calculates the result of the affine function f(x) = W*X + b,
        then applies the layer's activation function to the results.
        Arguments:
            input (np.ndarray): the layer's inputs
        Returns:
            The layer's calculated output in the form of a numpy array.
        """
        affine = np.dot(input, self.weights) + self.biases
        layer_output = self._apply_activation(affine)
        return layer_output

    def backward(self, input, output_gradient):
        """
        Computes the gradient that the current layer will send to
        the previous layer for the back-propagation process, then
        updates the current layer's weights and biases according to
        the gradient backpropagated by the next layer.
        Arguments:
            input (np.ndarray): the layer's inputs
            output_gradient (np.ndarray): received from the next layer
        Returns:
            This layer's calculated gradient to be sent to the previous
            layer
        """
        # Compute the gradient of the activation function of the current layer
        affine = np.dot(input, self.weights) + self.biases
        current_gradient = self._compute_activation_gradient(affine,
                                                             output_gradient)

        # Compute the current layer's gradient to be sent to the previous layer
        input_gradient = np.dot(current_gradient, self.weights.T)

        # Update current layer's weights
        weights_diff = np.dot(input.T, current_gradient)
        biases_diff = current_gradient.mean(axis=0) * input.shape[0]
        self.weights = self.weights - self.learning_rate * weights_diff
        self.biases = self.biases - self.learning_rate * biases_diff
        return input_gradient

    def _generate_weights(self, input_size, output_size):
        """
        Generate weights according to the input size, output size, and
        the specified weight initializer. Methods are chosen between He,
        Xavier and a 'vanilla' random initialisation, and each of them can
        use a normal or uniform repartition.
        Arguments:
            input_size (int): size of the layer's input units
            output_size (int): size of the layer's output units
        Returns:
            The weights in a numpy array
        """
        match self.weights_initializer:
            case 'heUniform' | 'he_uniform':
                weights = wi.he_uniform((input_size, output_size))
            case 'heNormal' | 'he_normal':
                weights = wi.he_normal((input_size, output_size))
            case 'xavierUniform' | 'xavier_uniform':
                weights = wi.xavier_uniform((input_size, output_size))
            case 'xavierNormal' | 'xavier_normal':
                weights = wi.xavier_normal((input_size, output_size))
            case 'randomUniform' | 'random_uniform':
                weights = wi.random_uniform((input_size, output_size))
            case 'randomNormal' | 'random_normal':
                weights = wi.random_normal((input_size, output_size))
            case _:
                weights = wi.random_uniform((input_size, output_size))
        return weights

    def _apply_activation(self, affine_result):
        """
        Applies the layer's activation function to the result of the
        affine transformation f(x) = W*X + b
        Arguments:
            affine_result (np.ndarray): the result of the affine transformation
        Returns:
            A numpy array containing the output of the layer
        """
        match self.activation:
            case 'sigmoid':
                output = af.sigmoid(affine_result)
                self.cache = output
            case 'relu':
                output = af.relu(affine_result)
            case 'tanh':
                output = af.tanh(affine_result)
                self.cache = output
            case 'softmax':
                output = af.softmax(affine_result)
            case None:
                output = affine_result
            case _:
                output = af.sigmoid(affine_result)
        return output

    def _compute_activation_gradient(self, x, output_gradient):
        """
        Calculates the layer's activation function's gradient as an
        intermediary step to compute the whole gradient of the layer
        Arguments:
            x (np.ndarray): the layer's input
            output_gradient (np.ndarray): the next layer's gradient
        Returns:
            A numpy array representing the activation function's gradient
        """
        match self.activation:
            case 'sigmoid':
                output = af.sigmoid_gradient(x, output_gradient, self.cache)
            case 'relu':
                output = af.relu_gradient(x, output_gradient)
            case 'tanh':
                output = af.tanh_gradient(x, output_gradient, self.cache)
            case 'softmax':
                output = output_gradient
            case _:
                output = output_gradient
        return output

    def __str__(self):
        """
        Called when printing an instance of this class
        """
        layer_to_string = f"{self.weights.shape[1]} neurons, "
        layer_to_string += f"{self.activation} activation function, "
        layer_to_string += f"weights initialised using "
        layer_to_string += f"{self.weights_initializer}."
        return layer_to_string
