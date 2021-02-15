"""
This Script holds common Neuron Activation Functions & their derivatives
"""

import numpy as np

def sigmoid(z):
    """
    Sigmodial Nonlinear Activation Function
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """
    Returns Derivative of Sigmodial Nonlinear Activation Function
    """
    sigmoid_val = sigmoid(z)
    return sigmoid_val * (1 - sigmoid_val)

def tangent_hyperbolic(z):
    """
    Hyperbolic Tangent Nonlinear Activation Function
    """
    return np.tanh(z)

def tangent_hyperbolic_prime(z):
    """
    Returns Derivative of Tangent Nonlinear Activation Function
    """
    tanh_val = tangent_hyperbolic(z)
    return 1-(tanh_val)**2


def rectified_linear(z):
    """
    ReLU Nonlinear Activation Function
    """
    return max(0, z)

def rectified_linear_prime(z):
    """
    Returns Derivative of ReLU Nonlinear Activation Function
    """
    relu_val = rectified_linear(z)
    if relu_val<=0:
        return 0
    else:
        return 1