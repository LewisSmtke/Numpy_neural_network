"""Module containing activation functions and their derivative"""
import numpy as np


def relu(input_vector):
    """
    Function used for applying the relu activation function to each element in a given input_vector.
    It returns the input value if the input value is greater than 0, otherwise returns 0.
    """
    relu_vector = np.maximum(0, input_vector)
    return relu_vector


def relu_deriv(input_vector):
    """
    Function used for calculating the derivative of the relu function.
    Strictly speaking, relu activation doesn't have a derivative,as it is undefined in x=0
    But as for convention, we assume f'(x=0)=0
    """
    # np.heaviside(x1,x2): returns 0 if x1 < 0, returns x2 if x1 == 0 and returns 1 if x1 > 0
    relu_deriv_vector = np.heaviside(input_vector, 0)
    return relu_deriv_vector


def softmax(input_vector):
    """
    Function used for applying the softmax activation to each element in a given input_vector.
    Softmax (i) =  e^z(i) / sum(e^z(j)) | sum from j=1 over N
    """
    # The values are NOT probabilities, but rather certainties
    softmax_vector = np.exp(input_vector) / np.sum(np.exp(input_vector))
    return softmax_vector


def tanh(input_vector):
    """
    Function used for applying the tanh activation to each element in a given input_vector.
    """
    tanh_vector = np.tanh(input_vector)
    return tanh_vector


def tanh_deriv(input_vector):
    """
    Function used for calculating the derivative of the tanh function.
    """
    # tanh'(x) = 1 - tanh(x)^2
    tanh_deriv_vector = 1 - (tanh(input_vector)) ** 2
    return tanh_deriv_vector


def sigmoid(input_vector):
    """
    Function used for applying the sigmoid activation to each element of a given input_vector.
    sigmoid(z) = 1 / 1 + exp(-z)
    """
    sigmoid_vector = 1 / (1 + np.exp(-input_vector))
    return sigmoid_vector
