"""
This script contains the Neuron Class for utility in our FC ANN
"""

# making necessary imports
import numpy as np


class Neuron:
    """
    Instance Variables:
        W --> Weight Row Vector [w1,w2,w3...wn, bias]
        X --> Input Row Vector [x1,x2,x3,...xn,1]
        Activation Function Choice
            Sigmoid --> Sigmodial
            TanHyp --> Tangent Hyperbolic
            Relu --> Rectified Linear

    Key Methods:
        Output:
            This Returns the Numeric Output of our neuron after it dots our Weight & Input Vectors (W & X)
            and applies the nonlinear activation function
        Adjust Params
            Once Implemented, this method will adjust our parameters during the backpropagation process
    """

    def __init__(self, X, W, activation_func):
        """
        Constructor for our Neuron
        X = complete input vector w/ bias complement of 1 added as last term
        W = complete weight vector with bias as last term
        """
        self.X = X
        self.W = W
        if activation_func.lower() == 'sigmoid':
            self.activation_func = 'sigmoid'
        elif activation_func.lower() == 'tanhyp':
            self.activation_func = 'tangent_hyperbolic'
        elif activation_func.lower() == 'relu':
            self.activation_func = 'rectified_linear'
        else:
            raise Exception("Must Provide a Valid Activation Function (Sigmodial, Hyperbolic Tangent, ReLU) for Neuron")

    def output(self):
        """
        Returns our numeric neuron output after applying appropriate activation_func
        """
        # z represents our complete neuron input of x1w1 + x2w2 + ... xnwn + bias
        z = np.dot(self.X, self.W)

        # applying appropriate non-linear activation function & returning neuron output
        if self.activation_func == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation_func == 'tangent_hyperbolic':
            return self.tangent_hyperbolic(z)
        elif self.activation_func == 'rectified_linear':
            return self.rectified_linear(z)

    def sigmoid(self, z):
        """
        Sigmodial Nonlinear Activation Function
        """
        return 1 / (1 + np.exp(-z))

    def tangent_hyperbolic(self, z):
        """
        Hyperbolic Tangent Nonlinear Activation Function
        """
        return np.tanh(z)

    def rectified_linear(self, z):
        """
        ReLU Nonlinear Activation Function
        """
        return max(0, z)

    def update_params(self):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}(Input Vector: {self.X}, Weight Vector: {self.W}, ' \
               f'Activation Function: {self.activation_func})'

    def __str__(self):
        return f'Neron with Inputs: {self.X} | Weights: {self.W} | Activation Function: {self.activation_func}'


if __name__ == '__main__':
    # testing our neuron class
    inputs = np.array([1, 2, 3, 1])
    weights = np.array([1, 5, 10, 0])
    act_func = 'sigmoid'
    n1 = Neuron(inputs, weights, act_func)
    print(n1)
    print(n1.output())
