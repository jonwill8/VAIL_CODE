"""
This script contains the Macro Network Class for our fully connected ANN
This ANN will perform both Binary classification (log loss GD optimization) and regression (MSE loss GD optimization)
"""

#making nessacary imports
from Layer import Layer
import numpy as np
import math

class Network:
    """
    Instance Variables
    Cost Function Selection --> MSE or Log Loss
    x train Data --> must be a single row/column vector
    y Train Data --> must be a single row/column vector
        NOTE: in constructor, concatenate the x/y train data into a single np array to aid w/ shuffling
    x test Data --> must be a single row/column vector
    y test Data --> must be a single row/column vector
    split_index --> the index we use to split x/y training data in our train data np array
    epoch_num --> number of epochs we train our model
    self.network --> Our initial empty np array of Layers
    batch_size --> the batch size we will split our training data into
        ex: batch size of 3 means splitting train data into thirds
            will have to use floor of x train Data/batch size
                on last batch will take last row index --> rest of array
        note: this batch_size must be less than 1/2 of x data size
    learning rate
    layer_num --> number of layers this ANN will have (helper for add layer method)
    note: self.predictions is a column array which is dynamiclly recreated every batch
    """
    def __init__(self,cost_func,x_train,y_train,x_test,y_test,epoch_num,batch_size,layer_num,learn_rate):
        #init appropriate IVs
        if cost_func.lower() not in('mse','log-loss'):
            raise Exception('Must Provide a Valid Model Cost Function')
        else:
            self.cost_func = cost_func.lower()
        #creating np array which holds both x_train & associated y train
        #note: you split the train_data np array at the self.split index
            #x_arr = self.train_data[:,:self.train_split_index]
            #y_arr = self.train_data[:,train_split_index:]
        self.train_split_index = x_train.shape[1]
        self.train_data = np.append(x_train,y_train,axis=1)
        self.x_test = x_test
        self.y_test = y_test
        self.network = np.empty(layer_num,dtype=object)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.learn_rate = learn_rate

    def add_Layer(self,neuron_number,neuron_activation_func,layer_type,next_layer_depth):
        """
        Adds a Layer object to our self.network np array
        Note: Set next_layer_depth = to 1 for output layer
        """
        #init the index to which we will drop our created Layer Object
        add_index = np.count_nonzero(self.network)

        #calculating the fan_in to this new layer
        if layer_type.lower() == 'input':
            #fan_in is simply the number of x input variables w have for input layer
            fan_in = self.train_split_index
        else:
            #fan_in for hidden/output layer is simply the depth (neuron #) of the previous layer
            fan_in = self.network[-1].neuron_number

        #calculating the fan_our of this new layer
        if layer_type.lower() == 'output':
            fan_out = neuron_number
        else:
            fan_out = next_layer_depth

        #creating layer object & adding to self.network
        layer = Layer(neuron_number,neuron_activation_func,layer_type,fan_in,fan_out)
        self.network[add_index] = layer

    def forward_propogate(self,x_train_subset):
        """
        This method feeds x input vector through model for n times
        & appends our model prediction to an np array (self.predictions)
        NOTE:
        input matrix:
        [[input vec 1]
         [input vec 2]
         [input vec 3]
         ...
         [input vec n]
                      ]
        the actual input matrix is a subslice of our entire training data
        (we will handle the subslicing use batch numbers and an index var in train function)
        """
        #defining function to recursively pull the output of the last layer
        def recur_output(layer_index,input_vector):
            if layer_index == 0: #we are at the input layer
                #return the output vector from our input layer being feed the complete input data vector
                return self.network[0].output_vector(input_vector)
            else:
                return self.network[layer_index].output_vector(recur_output(layer_index-1,input_vector))

        #defining output vector.
        #The outut_vec has the # of rows as our x_train_subset matrix
        #The output_vec has the # of columns as out y_train/y_test matrix
        output_vec = np.zeros((x_train_subset.shape[0],self.y_test.shape[1]))
        #calculating model output vector and keeping track in matri
        for row_index in range(x_train_subset.shape[0]):
            #pulling the ith input vector from our x_train subset matrix
            curr_input_vec = x_train_subset[row_index,:]
            #pulling the model output w.r.t the ith input vector
            curr_output_vec = recur_output(self.layer_num-1,input_vector)
            #appending our model output prediction to our output vector
            output_vec[row_index,:] = curr_output_vec


if __name__ == '__main__':
    pass
