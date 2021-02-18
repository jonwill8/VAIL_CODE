"""
This script contains the Neuron Class for utility in our FC ANN
"""

# making necessary imports
import numpy as np
import Neuron_Math as nm

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

    def __init__(self, activation_func):
        if activation_func.lower() == 'sigmoid':
            self.activation_func = 'sigmoid'
        elif activation_func.lower() == 'tanhyp':
            self.activation_func = 'tangent_hyperbolic'
        elif activation_func.lower() == 'relu':
            self.activation_func = 'rectified_linear'
        else:
            raise Exception("Must Provide a Valid Activation Function (Sigmodial, Hyperbolic Tangent, ReLU) for Neuron")

    def output(self,X):
        """
        Returns our numeric neuron output after applying appropriate activation_func
        X = complete input vector w/ bias complement of 1 added as last term
        W = complete weight vector with bias as last term
        """
        self.X = X

        # z represents our complete neuron input of x1w1 + x2w2 + ... xnwn + bias
        z = np.dot(self.X, self.W)
        self.Z = z

        # applying appropriate non-linear activation function & returning neuron output
        if self.activation_func == 'sigmoid':
            return nm.sigmoid(z)
        elif self.activation_func == 'tangent_hyperbolic':
            return nm.tangent_hyperbolic(z)
        elif self.activation_func == 'rectified_linear':
            return nm.rectified_linear(z)

    def init_weights(self,W):
        """
        This function initilizes our model weights to the provided W vector
        """
        self.W = W

    def update_weights(self,learn_rate,partial_wrt_weight):
        self.W = self.W - learn_rate*partial_wrt_weight

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
    n1 = Neuron(act_func)
    n1.init_weights(weights)
    print(n1.output(inputs))
    print(n1)