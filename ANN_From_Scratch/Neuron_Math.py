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
    zero_arr = np.zeros(z.shape)
    return np.maximum(zero_arr,z)

def rectified_linear_prime(z):
    """
    Returns Derivative of ReLU Nonlinear Activation Function
    """
    z[z>0]=1
    z[z<=0]=0
    return z