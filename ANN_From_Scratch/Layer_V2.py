"""
This Script contains the 2nd iteration of our Layer Class
"""

#making nessacary imports
from Neuron import Neuron
import numpy as np
import Neuron_Math as nm
import math

class Layer2:
    def __init__(self,neuron_number,neuron_activation_func,layer_type,fan_in,fan_out):
        # init appropriate IVs
        self.neuron_number = neuron_number
        self.neuron_activation_func = neuron_activation_func.lower()
        self.layer_type = layer_type.lower()
        self.fan_in = fan_in
        self.fan_out = fan_out

        # raising exceptions if an invalid layer type  passed
        if layer_type.lower() not in ('initial_hidden', 'hidden', 'output'):
            raise Exception('Enter a Valid Type for This Layer')

        #raising exception if an invalid activation function is chosen
        if self.neuron_activation_func.lower() not in ('sigmoid', 'tanhyp', 'relu'):
            raise Exception('Enter a Valid Type for The Neurons of this Layer')

        #init our 2d matrx of neuron weights
        scaler = None
        if self.neuron_activation_func == 'sigmoid':
            scaler = math.sqrt(2 / (self.fan_in + self.fan_out))
        elif self.neuron_activation_func == 'tanhyp':
            scaler = math.sqrt(1 / self.fan_in)
        elif self.neuron_activation_func == 'relu':
            scaler = math.sqrt(2 / self.fan_in)
        self.neuron_weights = np.array([ [np.random.normal(scale=scaler)]*fan_in for _ in range(self.neuron_number)])
        #init our column vector of neuron biases
        self.neuron_biases = np.array([[1] for _ in range(self.neuron_number)])

    def gen_output_vector(self,input_vec):
        """
        This method returns the column output vector of our Layer provided a Column Input Vector
        """
        #raising exception if not fed a column vector of data
        if input_vec.shape[1] != 1:
            raise Exception('Layers Can only be feed input data contained in a column np array (dimensons nx1)')
        #raising exception if our input data dimensions doesn't align with our weight #
        if self.neuron_weights.shape[1] != input_vec.shape[0]:
            raise Exception(f'This Neuron must be provided a Column Vector of '
                            f'Length {self.neuron_weights.shape[1]} not {input_vec.shape[0]}')
        #holding our layer's Z value (weight dot input + bias) in an IV
        self.Z = np.dot(self.neuron_weights,input_vec) + self.neuron_biases
        #holding our layers pushed output in an IV & returning the output for processing in forward prop
        if self.neuron_activation_func == 'sigmoid':
            self.output_vector = nm.sigmoid(self.Z)
        elif self.neuron_activation_func == 'tanhyp':
            self.output_vector = nm.tangent_hyperbolic(self.Z)
        elif self.neuron_activation_func == 'relu':
            self.output_vector = nm.rectified_linear(self.Z)
        #returning the output vector
        return self.output_vector

    def update_weights_output_layer(self,learn_rate,error_vec,last_layer_output):
        """
        This method updates our output layer neuron weights provided a learning rate & our returned self.error vector
        """
        #checking if we should transpose last layer output vector
        partial_cost_wrt_weights = last_layer_output.T*error_vec
        #checking dimensions of our partial_cost_wrt_weights matrix
        if partial_cost_wrt_weights.shape[1]!= self.neuron_weights.shape[1]:
            raise Exception (f'The Col # of the Partial Error Weight Update matrix does not match the column '
                             f'# of this output layer\'s neuron weight matrix. You must have equivlanet dimensions '
                             f'to perform backprop!')
        else:
            self.log_old_weights()  # logging the old weight values to use in backprop
            self.neuron_weights = self.neuron_weights - learn_rate*partial_cost_wrt_weights

    def update_weights_hidden_layer(self,learn_rate,error_vec,last_layer_output):
        """
        This method updates our neuron hidden weights provided a learning rate & our returned self.error vector
        Note: check if this method has bad dimensionlaity for matrix operations!
        """
        #perhaps np.sum(layer+1 weights transposed)
        #reting if we should transpose last layer output vector
        partial_cost_wrt_weights = last_layer_output*error_vec
        #checking dimensions of our partial_cost_wrt_weights matrix
        if partial_cost_wrt_weights.shape != self.neuron_weights.shape:
            raise Exception (f'The Partial Error Weight Update matrix of dimensons {partial_cost_wrt_weights.shape} '
                             f'does not match the dimensons of this layer\'s neuron weight matrix: {self.neuron_weights.shape}')
        else:
            self.log_old_weights()  # logging the old weight values to use in backprop
            self.neuron_weights = self.neuron_weights - learn_rate * partial_cost_wrt_weights



    def update_biases(self,learn_rate,error_vec):
        """
        This method updates our biases vector during backpropogation.
        error_vec = the result of calling calculate hidden/output error on THIS layer
        Must call calculate_error method first then update biases!
        """
        if error_vec.shape != self.neuron_biases.shape:
            raise Exception (f'The Layer Error Bias Update Vector of dimensons {error_vec.shape.shape} '
                             f'does not match the dimensons of this layer\'s neuron bias matrix: {self.neuron_biases.shape}')
        else:
            self.neuron_biases = self.neuron_biases - learn_rate*error_vec


    def calculate_output_layer_error(self,error_prime_vec):
        """
        this method calculates and sets as an IV the output layer's error vector
        """
        #ensuring uniform dimensons for hagmard product

        if error_prime_vec.shape != self.Z.shape:
            raise Exception(f'The passed Error Prime Vector with dimensons {error_prime_vec.shape} '
                            f'must have the same dimensons as this output Layer\'s Z prime Vector {self.Z.shape} ')
        else:
            #calculating the proper z prime vector (we have uniform dims)
            Z_prime = None
            if self.neuron_activation_func == 'sigmoid':
                Z_prime = nm.sigmoid_prime(self.Z)
            elif self.neuron_activation_func == 'tanhyp':
                Z_prime = nm.tangent_hyperbolic_prime(self.Z)
            elif self.neuron_activation_func == 'relu':
                Z_prime = nm.rectified_linear_prime(self.Z)

            self.error_vec = error_prime_vec*Z_prime
            return self.error_vec

    def calculate_hidden_layer_error(self,next_layer_weights,next_layer_error_vec):
        """
        this method calculates and sets as an IV a hidden layer's error vector
        next_layer_weights is transposed after it is passed
        """
        # calculating the proper z prime vector
        Z_prime = None
        if self.neuron_activation_func == 'sigmoid':
            Z_prime = nm.sigmoid_prime(self.Z)
        elif self.neuron_activation_func == 'tanhyp':
            Z_prime = nm.tangent_hyperbolic_prime(self.Z)
        elif self.neuron_activation_func == 'relu':
            Z_prime = nm.rectified_linear_prime(self.Z)

        self.error_vec = (next_layer_weights.T*next_layer_error_vec)*Z_prime
        return self.error_vec


    def log_old_weights(self):
        
        #this helper method logs our layer's neuron weight values before updating during backprop
        
        self.old_weights = self.neuron_weights



    def __repr__(self):
        if self.layer_type == 'output':
            return f'{self.layer_type.title()} Layer with {self.neuron_number} {self.neuron_activation_func} Comprising Neurons. \n' \
                   f'Neuron Weights Array: \n {self.neuron_weights}'
        else:
            return f'{self.layer_type.title()} Layer with {self.neuron_number} {self.neuron_activation_func} Comprising Neurons. \n' \
                   f'Neuron Weights Array: \n {self.neuron_weights} \n Next Layer in the NN is {self.fan_out} Neurons Deep '