"""
This script contains the Macro Network Class for our fully connected ANN
This ANN will perform both Binary classification (log loss GD optimization) and regression (MSE loss GD optimization)
"""


#making nessacary imports
from Layer_V2 import Layer2
import numpy as np
import math
import matplotlib.pyplot as plt

#TODO Implement Parameter Regularization Technqiues (ex: Neuron Dropout)
#TODO TRACE BACKPROPOGATION METHOD. I SUSPECT WEIGHT MATRICES ARE CHANGING 
""" 
If the NN is a regressor, then the output layer has a single node.

If the NN is a classifier, then it also has a single node unless softmax is used in which case the output layer has one node per class label in your model.
"""
class Network:

    def __init__(self,cost_func,x_train,y_train,x_test,y_test,epoch_num,batch_size,layer_num,layer_depths,learn_rate):
        #init appropriate IVs
        if cost_func.lower() not in('mse','log-loss'):
            raise Exception('Must Provide a Valid Model Cost Function')
        else:
            self.cost_func = cost_func.lower()
        #check to ensure all data is in 2d vectors
        if x_train.ndim==1 or y_train.ndim==1 or x_test.ndim==1 or y_test.ndim==1:
            raise Exception('Passed Train & Tese Data must be wrapped in a 2d numpy array!')
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
        self.layer_depths = layer_depths
        self.learn_rate = learn_rate
        self.error_log = np.zeros((self.epoch_num*self.batch_size))

    def add_Layer(self,neuron_activation_func,layer_type):
        """
        Adds a Layer object to our self.network np array
        Note: enusre that when adding layers the order is hidden input --> hidden_1 --> hidden_2 --> hidden_n --> output
        """
        #init the index to which we will drop our created Layer Object
        add_index = np.count_nonzero(self.network)
        #pulling the neuron # associated with this layer
        neuron_number = self.layer_depths[add_index]

        if add_index==0: #this is the initial_hidden layer
            # fan_in is simply the number of x input variables w have for input layer
            fan_in = self.train_split_index
        else: #this is a hidden/output layer
            fan_in = self.layer_depths[add_index-1]

        #calculating the fan_out of this new layer
        if layer_type.lower() == 'output':
            fan_out = neuron_number
        else:
            fan_out = self.layer_depths[add_index+1]

        #creating layer object & adding to self.network
        layer = Layer2(neuron_number,neuron_activation_func,layer_type,fan_in,fan_out)
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
        """
        #defining function to recursively pull the output of the last layer
        def recur_output(layer_index,input_vector):
            if layer_index == 0: #we are at the initial hidden layer
                #return the output vector from our input layer being feed the complete input data vector
                return self.network[0].gen_output_vector(input_vector)
            else:
                return self.network[layer_index].gen_output_vector(recur_output(layer_index-1,input_vector))

        #defining output vector.
        #The outut_vec has the # of rows as our x_train_subset matrix
        #The output_vec has the # of columns as out y_train/y_test matrix
        output_vec = np.zeros((x_train_subset.shape[1],self.y_test.shape[1]))
        #calculating model output vector and keeping track in matrix
        for index in range(x_train_subset.shape[1]):
            #pulling the ith input vector from our x_train subset matrix.
            # Note: we tarnspose becuase column slice auto reverts to a row vector
            curr_input_vec = np.array([x_train_subset[:,index]]).T
            #pulling the model output w.r.t the ith input vector
            curr_output_vec = recur_output(self.layer_num-1,curr_input_vec)
            #appending our model output prediction to our output vector
            output_vec[index,:] = curr_output_vec.T

        #returning our matrix of model predictions
        return output_vec

    def train(self):
        """
        This method trains our model to the x/y training data.
        """
        # init the index counter for our error log
        error_log_index = -1
        for epoch in range(self.epoch_num):
            #shuddling training data
            np.random.shuffle(self.train_data)
            if self.train_data.shape[0] % self.batch_size != 0:
                #cheking to make sure batches can be split even;y
                raise Exception('Batch_size must evenly divide training data!')
            else:
                for sub_data in np.split(self.train_data, self.batch_size):
                    #incrementing error_log
                    error_log_index+=1
                    #spltting subdata into x input vector and associated y output  vector
                    x_train_subset,y_train_subset = sub_data[:,:self.train_split_index], sub_data[:,self.train_split_index:]

                    #pulling our model's prediciton matrix on the x_train_subset data
                    prediction_matrix = np.array([self.forward_propogate(x_train_subset.T)])

                    #printing model error every prediction batch
                    if self.cost_func == 'mse':
                        print_error = self.mean_squared_error(prediction_matrix,y_train_subset)
                    elif self.cost_func == 'log-loss':
                        print_error = self.log_loss_error(prediction_matrix, y_train_subset)

                    #print(f'Current Model Error {print_error}')
                    #adding current model error to log
                    self.error_log[error_log_index] = print_error

                    #updating params via graident descent + backpropogation
                    self.backprop(x_train_subset,prediction_matrix,y_train_subset)

           #printing epoch number
            print(f'Epoch # {epoch+1}')


    def backprop(self,x_inputs_vector,y_predictions_vector,y_observation_vector):

        #iterating backwards through the self.network Layer np array
        for index in range(len(self.network)-1,-1,-1):

            #pulling layers in reverse
            curr_layer = self.network[index]
            if curr_layer.layer_type == 'output':
                output_neruon_num = curr_layer.neuron_number
                error_prime_vec = self.mse_prime(y_predictions_vector,y_observation_vector,output_neruon_num)
                output_layer_error = curr_layer.calculate_output_layer_error(error_prime_vec)
                #updating weights/biases:
                curr_layer.update_biases(self.learn_rate,output_layer_error)
                curr_layer.update_weights_output_layer(self.learn_rate,output_layer_error,self.network[index-1].output_vector)
            if curr_layer.layer_type == 'hidden':
                next_layer_weights = self.network[index+1].old_weights
                next_layer_error =  self.network[index+1].error_vec
                hidden_layer_error = curr_layer.calculate_hidden_layer_error(next_layer_weights,next_layer_error)
                #updating our bias
                curr_layer.update_biases(self.learn_rate,hidden_layer_error)
                #pulling nessacary input vector and then updating weights
                prev_layer_output =  self.network[index-1].output_vector
                curr_layer.update_weights_hidden_layer(self.learn_rate,hidden_layer_error,prev_layer_output)
            if curr_layer.layer_type == 'initial_hidden':
                next_layer_weights = self.network[index + 1].old_weights
                next_layer_error = self.network[index + 1].error_vec
                hidden_layer_error = curr_layer.calculate_hidden_layer_error(next_layer_weights, next_layer_error)
                # updating our bias
                curr_layer.update_biases(self.learn_rate, hidden_layer_error)
                # pulling nessacary input vector and then updating weights
                prev_layer_output = x_inputs_vector
                curr_layer.update_weights_hidden_layer(self.learn_rate, hidden_layer_error, prev_layer_output)

        #resetting each layer's old_weights & errors IV to Nones
        for layer in self.network:
            layer.old_weights = None
            layer.error_vec = None

    def mean_squared_error(self,y_predictions_vector,y_observation_vector):
        """
        This function returns the means squared error associated with our prediction vector
        Used for printing how our model is fitting to the data each epoc (regression)
        """
        return 1/self.batch_size*np.sum((y_observation_vector-y_predictions_vector)**2)

    def mse_prime(self,y_predictions_vector,y_observation_vector,output_layer_depth):
        """
        This function returns the derivative of mse w.r.t the y_predictions_vector. Used in backprop
        We stack our average errors w.r.t the number of neurons in our output layer
        """
        #note: may have to altert the # of colums returned from this func if we have 2+ nodes in output layer
        #code may possibly become:

        """
        return np.array([ [-2/self.batch_size*np.sum((y_observation_vector-y_predictions_vector)**2)]*self.y_test.shape[1]
                          for _ in range(output_layer_depth) ])
        """
        return np.array([ [-2/self.batch_size*np.sum((y_observation_vector-y_predictions_vector)**2)]
                          for _ in range(output_layer_depth) ])

    def log_loss_error(self,y_predictions_vector,y_observation_vector):
        """
        This function returns the log loss error associated with our prediction vector
        Used for printing how our model is fitting to the data each epoc (classification)
        """
        return -1/self.batch_size*np.sum(y_observation_vector*np.log(y_predictions_vector)+(1-y_observation_vector)*np.log(1-y_predictions_vector))

    def log_loss_prime(self,y_predictions_vector,y_observation_vector):
        """
        This function returns the derivative of log loss w.r.t the y_predictions_vector. Used in backprop
        """
        return np.array([ [1/self.batch_size*np.sum(-(y_observation_vector/y_predictions_vector)+(1-y_observation_vector)/(1-y_predictions_vector))]
                          for _ in range(output_layer_depth) ])


    def plot_train_error(self):
        """
        this method plots our error while training
        """
        plt.plot(np.arange(1,len(self.error_log)+1),self.error_log)
        plt.title('Model Error While')
        plt.xlabel('Training Batch')
        plt.ylabel('Model Error')
        plt.show()
        #printing error log to terminal
        print(self.error_log)


    def test_regression(self):
        """
        this method tests the regression capabilities of our model
        """
        model_predictions = self.forward_propogate(self.x_test.T)
        # calculating & displayinf RMSE
        RMSE = (self.mean_squared_error(model_predictions, self.y_test)) ** 0.5
        print(f'Regression Model has an RMSE accuracy of: {RMSE}')

    def test_classification(self):
        """
        this method tests the classification capabilities of our model
        """
        model_predictions = self.forward_propogate(self.x_test.T)
        right_predicts = 0
        for prediction , observation in zip(model_predictions,self.y_test):
            if observation==1 and 0.5<=prediction<=1:
                right_predicts+=1
            elif observation==0 and 0<=prediction<0.5:
                right_predicts+=1
            else:
                pass
        model_accuracy = right_predicts/model_predictions.shape[0] * 100
        print(f'The Binary Classification Model has an accuracy of {model_accuracy}%')

    def __repr__(self):
        return f'Network trained with {self.cost_func} Cost Function. Layers: \n' \
               f'{self.network}'


if __name__ == '__main__':
    #testing the forward propogation of a network which solves the XOR problem
    x_train = np.array([[0,0],[1,0],[0,1],[1,1]])
    x_train2 = x_train
    y_train = np.array([[0],[1],[1],[0]])
    y_train_2 = np.array([[0], [1], [1], [1]])
    x_test = np.array([[0,0],[0,0],[0,1],[0,1],[1,0],[1,0],[1,1]])
    y_test = np.array([[0],[0],[1],[1],[1],[1],[0]])
    y_test_2 =  np.array([[1],[0,0],[1,1],[1,1],[1,1],[1,1],[0,0]])
    model = Network('mse',x_train2,y_train_2,x_train2,y_train_2,epoch_num=500,batch_size=2,layer_num=2,layer_depths=[2,1],learn_rate=0.001)
    #adding hidden layers
    model.add_Layer('sigmoid','initial_hidden')
    #model.add_Layer('sigmoid', 'hidden')
    model.add_Layer('sigmoid', 'output')  #addding output layer
    print(model)
    #training model functionality
    model.train()
    model.plot_train_error()
    #testing model
    model.test_classification()