"""
This script contains the Macro Network Class for our fully connected ANN
This ANN will perform both Binary classification (log loss GD optimization) and regression (MSE loss GD optimization)
"""


#making nessacary imports
from Layer import Layer
import numpy as np
import math

#TODO Implement Parameter Regularization Technqiues (ex: Neuron Dropout)
#TODO Watch if bias optimizes!
#TODO Analyze the fllowing:
""" 
If the NN is a regressor, then the output layer has a single node.

If the NN is a classifier, then it also has a single node unless softmax is used in which case the output layer has one node per class label in your model.
"""
class Network:

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
            fan_in = self.network[add_index-1].neuron_number

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
        #calculating model output vector and keeping track in matrix
        for row_index in range(x_train_subset.shape[0]):
            #pulling the ith input vector from our x_train subset matrix
            curr_input_vec = x_train_subset[row_index,:]
            #pulling the model output w.r.t the ith input vector
            curr_output_vec = recur_output(self.layer_num-1,curr_input_vec)
            #appending our model output prediction to our output vector
            output_vec[row_index,:] = curr_output_vec

        #returning our matrix of model predictions
        return output_vec

    def train(self):
        """
        This method trains our model to the x/y training data.
        """
        for epoch in range(self.epoch_num):
            #shuddling training data
            np.random.shuffle(self.train_data)
            if self.train_data.shape[0] % self.batch_size != 0:
                #cheking to make sure batches can be split even;y
                raise Exception('Batch_size must evenly divide training data!')
            else:
                for sub_data in np.split(self.train_data, self.batch_size):
                    #spltting subdata into x input vector and associated y output  vector
                    x_train_subset,y_train_subset = sub_data[:,:self.train_split_index], sub_data[:,self.train_split_index:]

                    #pulling our model's prediciton matrix on the x_train_subset data
                    prediction_matrix = np.array([self.forward_propogate(x_train_subset)])

                    #printing model error every prediction batch
                    if self.cost_func == 'mse':
                        print_error = self.mean_squared_error(prediction_matrix,y_train_subset)
                    elif self.cost_func == 'log-loss':
                        print_error = self.log_loss_error(prediction_matrix, y_train_subset)
                    print(f'Current Model Error {print_error}')

                    #updating params via graident descent + backpropogation
                    self.backprop(x_train_subset,prediction_matrix,y_train_subset)

           #printing epoch number
            print(f'Epoch # {epoch+1}')


    def backprop(self,x_inputs_vector,y_predictions_vector,y_observation_vector):
        # init backpropogating error variable
        partial_error = 1

        #iterating backwards through the self.network Layer np array
        for index in range(len(self.network)-1,-1,-1):
            #pulling layers in reverse
            curr_layer = self.network[index]

            if curr_layer.layer_type == 'output':
                #finding partial w.r.t weights in the current layer

                #calculating average cost func prime value
                if self.cost_func == 'mse':
                    partial_error *= self.mse_prime(y_predictions_vector, y_observation_vector)
                elif self.cost_func == 'log-loss':
                    partial_error *= self.log_loss_prime(y_predictions_vector, y_observation_vector)

                #pulling the input vector which feed into the output layer
                curr_layer_input_vec = self.network[index-1].output_vec

                #scaling error by the curr_layer backprop z prime vector
                z_prime_vec = curr_layer.backpropogate()

                partial_error*=z_prime_vec.T

                # NOTE: before you multiply by the input vector to a layer, set a variable as the current
                # running error (this is partial for bias).

                #setting bias
                partial_bias = partial_error

                #recall: We change the error var b/c we need to pass back the running error vector
                partial_error_output = partial_error

                partial_error_output = partial_error_output*curr_layer_input_vec

                #appending partial bias to end of partial error vector
                partial_error_output = np.append(partial_error_output,partial_bias,axis=1)

                #updating params
                curr_layer.update_weights(self.learn_rate,partial_error_output)

            elif curr_layer.layer_type == 'hidden':
                #pulling the input vector which fed into this hidden layer
                curr_layer_input_vec = self.network[index-1].output_vec

                #pulling the old weights of the proceeding layer
                next_layer_old_weights =  self.network[index+1].old_weights

                #pulling the z_prime vec associate with this layer
                z_prime_vec = curr_layer.backpropogate()

                #updating the partial error accordingly
                partial_error = partial_error.T*next_layer_old_weights
                partial_error *= z_prime_vec.T

                # holding our partial bias
                partial_bias = partial_error

                #switching var to preserve the passing back of the running error vector
                partial_error_hidden = partial_error

                #multiplying our errors by associated input vector to the layer
                partial_error_hidden*=curr_layer_input_vec

                #adding bias partial back in
                partial_error_hidden = np.append(partial_error_hidden,partial_bias,axis=1)

                # updating params
                curr_layer.update_weights(self.learn_rate,partial_error_hidden)

            elif curr_layer.layer_type == 'input':
                #pulling the input vector which fed into this input layer (the x data)
                curr_layer_input_vec = x_inputs_vector

                # pulling the old weights of the proceeding layer
                next_layer_old_weights = self.network[index + 1].old_weights

                # pulling the z_prime vec associate with this layer
                z_prime_vec = curr_layer.backpropogate()

                # updating the partial error accordingly (we do not actually have to sub a var b/c this is the last layer)

                partial_error = partial_error.T*next_layer_old_weights

                partial_error*=z_prime_vec.T

                #holding the bias
                partial_bias = partial_error

                partial_error_input = partial_error

                partial_error_input = partial_error_input.T*curr_layer_input_vec

                #adding bias back in
                partial_error_input = np.append(partial_error_input,partial_bias.T,axis=1)

                # updating params
                curr_layer.update_weights(self.learn_rate, partial_error_input)

        #resetting each layer's old_weights IV to Nones
        for layer in self.network:
            layer.old_weights = None


    def mean_squared_error(self,y_predictions_vector,y_observation_vector):
        """
        This function returns the means squared error associated with our prediction vector
        Used for printing how our model is fitting to the data each epoc (regression)
        """
        return 1/self.batch_size*np.sum((y_observation_vector-y_predictions_vector)**2)

    def mse_prime(self,y_predictions_vector,y_observation_vector):
        """
        This function returns the derivative of mse w.r.t the y_predictions_vector. Used in backprop
        """
        return -2/self.batch_size*np.sum((y_observation_vector-y_predictions_vector)**2)

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
        return 1/self.batch_size*np.sum(-(y_observation_vector/y_predictions_vector)+(1-y_observation_vector)/(1-y_predictions_vector))

    def test(self):
        """
        This method tests our model against the test data & provides and accuracy heuristic
        """
        if self.cost_func =='mse': #this is a regression model, use RMSE
            model_predictions = self.forward_propogate(self.x_test)
            #calculating & displayinf RMSE
            RMSE = (self.mean_squared_error(model_predictions,self.y_test))**0.5
            print(f'Regression Model has an RMSE accuracy of: {RMSE}')

        elif self.cost_func =='log-liss': #this is a classifier model, use 0.5 prediction cutoff
            model_predictions = self.forward_propogate(self.x_test)
            right_predicts = 0
            for prediction , observation in zip(model_predictions,self.y_test):
                if observation==1 and 0.5<=prediction<=1:
                    right_predicts+=1
                elif observation==0 and 0<=prediction<0.5:
                    right_predicts+=1
                else:
                    pass
            model_accuracy = right_predicts/model_predictions.shape[0] * 100
            printf(f'The Classification Model has an accuracy of {model_accuracy}%')

    def __repr__(self):
        return f'Network trained with {self.cost_func} Cost Function. Layers: \n' \
               f'{self.network}'


if __name__ == '__main__':
    #testing the forward propogation of a network which solves the XOR problem
    x_train = np.array([[0,0],[1,0],[0,1],[1,1]])
    y_train = np.array([[0],[1],[1],[0]])
    x_test = np.array([[0,0],[0,0],[0,1],[0,1],[1,0],[1,0],[1,1]])
    y_test = np.array([[0],[0],[1],[1],[1],[1],[0]])
    model = Network('mse',x_train,y_train,x_test,y_test,epoch_num=1000,batch_size=2,layer_num=2,learn_rate=0.001)
    #adding input layer
    model.add_Layer(2,'sigmoid','input',1)
    #addding output layer
    model.add_Layer(1, 'sigmoid', 'output', 1)
    #training model functionality
    model.train()
    #testing model
    model.test()

