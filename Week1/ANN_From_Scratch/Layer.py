"""
This Script contains the Layer class for our FC ANN
"""
#TODO Figure out how if you need to vectorize entier Layer Object and Eliminate Neuron Class
#TODO Create Network Class
#TODO Implemnt both MSE Cost Function for Regression & Log Loss Cost Function for Classification
#TODO Derive and Iterative/Recursive Process to Update all Parameters in the Model During BackProp (Finding Partials)
#TODO Create Good for Gradient Descent Algo (Semi-Batch)
#TODO Experiment with enforcing Data Standardization

#making nessacary imports
from Neuron import Neuron
import numpy as np
import math

class Layer:
    """
    Instance Variables:
    Neuron Number --> Number of Neurons we want to include in the layer
    Neuron_activation_func --> type of Neurons we want to include in the layer
    Layer Type (Input/Hidden/Output)
    Input Vector --> A vector of all inputs for this layer
        NOTE: The input vector will be our direct training data if this is a Input Layer
            The train data must be a 1d vector!
        NOTE: The input vector will be an array of all previous layer predictions if this is a hidden/output layer
    Neuron Layer --> The acutal np array/layer of our initilized nodes

    Key Methods:
    Output_Vec:
        This returns a Vector of the Outputs from every Neuron in our Layer after
        each Neuron is feed all input data
            NOTE: This basic ANN is built with a Fully Connected Architecture
    Adjust Params:
        Once Implemented, this method will adjust our parameters during the backpopogation process
    """

    def __init__(self,neuron_number,neuron_activation_func,layer_type,input_vec,next_layer_depth):
        #init appropriate IVs
        self.neuron_number = neuron_number
        self.neuron_activation_func = neuron_activation_func
        self.layer_type = layer_type
        self.next_layer_depth = next_layer_depth

        # raising an exception if an invalid layer type is passed
        if layer_type.lower() not in ('input','hidden','output'):
            raise Exception('Enter a Valid Type for This Layer')
        else: #we were provided a valid layer type string
            # raising an exception if an invalid nueron type is passed
            if neuron_activation_func.lower() not in ('sigmoid', 'tanhyp', 'relu'):
                raise Exception('Enter a Valid Type for The Neurons of this Layer')
            else: #we passed all input string checks

                #init an empty np array to hold all neurons in the layer
                self.layer = np.empty(neuron_number,dtype=object)

                # pulling the raw number of input variables we have
                num_inputs = len(input_vec)

                #init our number of incoming/outgoing connections for each neuron in this layer
                fan_in = num_inputs
                fan_out = self.next_layer_depth

                #appending 1 to the end of input vector to serve as complement to Neuron Bias term in each Neuron
                input_vec = np.append(input_vec,1)

                for index in range(self.neuron_number):
                    #randomly initilizing weights for each constructed neuron w/ optimial strategy
                    scaler = None
                    if neuron_activation_func.lower() == 'sigmoid':
                        scaler = math.sqrt(2/(fan_in+fan_out))
                    elif neuron_activation_func.lower() == 'tanhyp':
                        scaler = math.sqrt(1/fan_in)
                    elif neuron_activation_func.lower() == 'relu':
                        scaler = math.sqrt(2/fan_in)

                    weights = np.array([np.random.normal(scale=scaler) for _ in range(num_inputs)])
                    weights = np.append(weights,0) #addding intial 0 bias term

                    #creating Neuron Object & adding to our np layer array
                    neuron = Neuron(input_vec,weights,neuron_activation_func)
                    self.layer[index] = neuron

    def output_vector(self):
        #returning numpy array of the prediction from each neuron in self.layer
        return np.array([neuron.output() for neuron in self.layer])

    def backpropogate(self):
        pass

    def __repr__(self):
        return f'{self.layer_type.title()} Layer with {self.neuron_number} {self.neuron_activation_func} Comprising Neurons. \n' \
               f'Neuron Numpy Array: \n {self.layer} \n Next Layer in the NN is only {self.next_layer_depth} Neurons Deep '

    def __str__(self):
        return f'{self.layer_type.title()} Layer with {self.neuron_number} {self.neuron_activation_func} Comprising Neurons. \n' \
               f'Neuron Numpy Array: \n {self.layer} \n Next Layer in the NN is only {self.next_layer_depth} Neurons Deep '

if __name__ == '__main__':
    neuron_number = 20
    neuron_activation_func = 'tanhyp'
    layer_type = 'hidden'
    input_vec = np.array([1,2,8,61,0.1,7,8,5]) #last layer had a depth of 8
    next_layer_depth = 10
    layer1 = Layer(neuron_number,neuron_activation_func,layer_type,input_vec,next_layer_depth)
    print(layer1)
    print(layer1.output_vector())
    #testing chaining layer outputs (this is how the ANN will forward propogate
    layer1_output = layer1.output_vector()
    layer2 = Layer(5,'sigmoid','output',layer1_output,0)
    print(layer2)
    print(layer2.output_vector())



