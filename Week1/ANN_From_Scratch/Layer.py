"""
This Script contains the Layer class for our FC ANN
"""
#TODO Experiment with enforcing Data Standardization

#making nessacary imports
from Neuron import Neuron
import numpy as np
import Neuron_Math as nm
import math

class Layer:
    """
    Instance Variables:
    Neuron Number --> Number of Neurons we want to include in the layer
    Neuron_activation_func --> type of Neurons we want to include in the layer
    Layer Type (Input/Hidden/Output)
    Neuron Layer --> The acutal np array/layer of our initilized nodes
    Fan_in --> Number of inputs being feed into the neurons of this layer from the previous layer
    Fan_out --> Number of neurons in next Layer

    Key Methods:
    Output_Vec:
        This returns a Vector of the Outputs from every Neuron in our Layer after
        each Neuron is feed input data
            NOTE: This basic ANN is built with a Fully Connected Architecture
    Backpropogate:
        Once Implemented, this method will adjust our parameters during the backpopogation process
    """

    def __init__(self,neuron_number,neuron_activation_func,layer_type,fan_in,fan_out):
        #init appropriate IVs
        self.neuron_number = neuron_number
        self.neuron_activation_func = neuron_activation_func
        self.layer_type = layer_type
        self.fan_in = fan_in
        self.fan_out = fan_out

        # raising exceptions if an invalid layer type/ invalid nueron type is passed
        if layer_type.lower() not in ('input','hidden','output'):
            raise Exception('Enter a Valid Type for This Layer')
        if self.neuron_activation_func.lower() not in ('sigmoid', 'tanhyp', 'relu'):
            raise Exception('Enter a Valid Type for The Neurons of this Layer')
        else: #we were provided a valid layer/Neuron data

            #init an empty np array to hold all neurons in the layer
            self.layer = np.empty(neuron_number,dtype=object)

            #init the each neuron
            for index in range(self.neuron_number):

                scaler = None
                if self.neuron_activation_func.lower() == 'sigmoid':
                    scaler = math.sqrt(2 / (self.fan_in + self.fan_out))
                elif self.neuron_activation_func.lower() == 'tanhyp':
                    scaler = math.sqrt(1 / self.fan_in)
                elif self.neuron_activation_func.lower() == 'relu':
                    scaler = math.sqrt(2 / self.fan_in)

                #initilizing weights
                weights = np.array([np.random.normal(scale=scaler) for _ in range(fan_in)])
                weights = np.append(weights, 0)  # addding intial 0 bias term

                #creating neuron obj
                neuron = Neuron(self.neuron_activation_func)
                #init neuron weights
                neuron.init_weights(weights)
                #adding neuron to our layer np array
                self.layer[index] = neuron


    def output_vector(self,input_vec):
        """
        This method returns our Layer Output Vector Provided a single row input vector
            ex: [x1,x2,x3,...xn,1] %1 added to complement each Neuron's bias term
        """
        input_vec = np.append(input_vec,1)
        self.output_vec = np.array([neuron.output(input_vec) for neuron in self.layer])
        return self.output_vec


    def backpropogate(self):
        """
        returns a vector which is the result of feeding the z vector to the derivative of our activation functions
        ** where the z vector in each row corresponds to the ith neuron's WX dot product
        """
        z_vec = np.array([neuron.Z for neuron in self.layer])
        if self.neuron_activation_func == 'sigmoid':
            z_vec = nm.sigmoid_prime(z_vec)
        elif self.neuron_activation_func == 'tangent_hyperbolic':
            z_vec =  nm.tangent_hyperbolic_prime(z_vec)
        elif self.neuron_activation_func == 'rectified_linear':
            z_vec = nm.rectified_linear_prime(z_vec)
        return z_vec



    def update_weights(self,learn_rate,partial_error):
        #updating each neuron with its new weights
        self.log_old_weights()
        for index,partial_error in enumerate(partial_error):
            #pulling ith neuron's partial error value
            partial_e = partial_error[index]
            #updating the value of the ith neuron weights
            self.layer[index].update_weights(learn_rate,partial_e)


    def log_old_weights(self):
        """
        this backpropogation helper method is used to create a temp IV of the layer's
        old neuron weight values before they are updated durng backprop
        """
        self.old_weights = np.array([nueron.W for neuron in self.layer])


    def __repr__(self):
        return f'{self.layer_type.title()} Layer with {self.neuron_number} {self.neuron_activation_func} Comprising Neurons. \n' \
               f'Neuron Numpy Array: \n {self.layer} \n Next Layer in the NN is {self.fan_out} Neurons Deep '

    def __str__(self):
        return f'{self.layer_type.title()} Layer with {self.neuron_number} {self.neuron_activation_func} Comprising Neurons. \n' \
               f'Neuron Numpy Array: \n {self.layer} \n Next Layer in the NN is {self.fan_out} Neurons Deep '

if __name__ == '__main__':
    neuron_number = 20
    neuron_activation_func = 'tanhyp'
    layer_type = 'hidden'
    input_vec = np.array([1,2,8,61,0.1,7,8,5]) #last layer had a depth of 8
    next_layer_depth = 10
    layer1 = Layer(neuron_number,neuron_activation_func,layer_type,fan_in=len(input_vec),fan_out=next_layer_depth)
    #print(layer1.output_vector(input_vec))
    #print(layer1)
    #testing chaining layer outputs (this is how the ANN will forward propogate
    layer2 = Layer(5,'sigmoid','output',fan_in=len(layer1.output_vector(input_vec)),fan_out=5)
    print(layer2.output_vector(layer1.output_vector(input_vec)))
    #print(layer2)



