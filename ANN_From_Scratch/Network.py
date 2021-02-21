"""
This script contains the Macro Network Class for our fully connected ANN
This ANN will perform both Binary classification (log loss GD optimization) and regression (MSE loss GD optimization)
"""

# TODO Implement Parameter Regularization Techniques (ex: Neuron Dropout)
# TODO TRACE BACKPROPAGATION METHOD. I SUSPECT WEIGHT MATRICES ARE CHANGING


# making necessary imports
from Layer_V2 import Layer2
import numpy as np
import matplotlib.pyplot as plt


class Network:

    def __init__(self, cost_func, x_train, y_train, x_test, y_test, x_feautures, epoch_num, batch_size, layer_num, layer_depths,
                 learn_rate):
        # init appropriate IVs
        if cost_func.lower() not in ('mse', 'log-loss'):
            raise Exception('Must Provide a Valid Model Cost Function')
        else:
            self.cost_func = cost_func.lower()

        # check to ensure all data is in 2d vectors
        if x_train.ndim == 1 or y_train.ndim == 1 or x_test.ndim == 1 or y_test.ndim == 1:
            raise Exception('Passed Train & Test Data must be wrapped in a 2d numpy array!')

        # creating np array which holds both x_train & associated y train
        # note: you split the train_data np array at the self.split index
        # x_arr = self.train_data[:,:self.train_split_index]
        # y_arr = self.train_data[:,train_split_index:]
        self.train_split_index = x_train.shape[1]
        self.train_data = np.append(x_train, y_train, axis=1)
        self.x_test = x_test
        self.y_test = y_test

        #init the numner of x features we have each sample
        self.x_feautures = x_feautures

        # creating our np array of layer objects
        self.network = np.empty(layer_num, dtype=object)

        # init all other IVs
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.layer_depths = layer_depths
        self.learn_rate = learn_rate
        self.error_log = np.zeros((self.epoch_num * self.batch_size))
        self.print_error = None

    def add_Layer(self, neuron_activation_func, layer_type):
        """
        Adds a Layer object to our self.network np array
        Note: ensure that when adding layers the order is hidden input --> hidden_1 --> hidden_2 --> hidden_n --> output
        """
        # init the index to which we will drop our created Layer Object
        add_index = np.count_nonzero(self.network)
        # pulling the neuron # associated with this layer
        neuron_number = self.layer_depths[add_index]

        if add_index == 0:  # this is the initial_hidden layer
            # fan_in is simply the number of x input variables w have for input layer
            fan_in = self.x_feautures #should math the n x variales we have tied to each y output
        else:  # this is a hidden/output layer
            fan_in = self.layer_depths[add_index - 1]

        # calculating the fan_out of this new layer
        if layer_type.lower() == 'output':
            fan_out = neuron_number
        else:
            fan_out = self.layer_depths[add_index + 1]

        # creating layer object & adding to self.network
        layer = Layer2(neuron_number, neuron_activation_func, layer_type, fan_in, fan_out)
        self.network[add_index] = layer

    def forward_propagation(self, x_train_subset):
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
        the actual input matrix is a sub-slice of our entire training data
        """

        # defining function to return model outputs for a provided x input col array
        def push_output(input_vector):
            # this function passes the initial input vector through the entire network.
            # Returns the prediction vector from our network's output layer
            output = input_vector
            for layer in self.network:
                output = layer.gen_output_vector(output)
            return output

        # defining output vector
        # The output_vec has the # of rows as our x_train_subset matrix
        # The output_vec has the # of columns as out y_train/y_test matrix

        output_vec = np.zeros((x_train_subset.shape[1], self.y_test.shape[1]))
        # calculating model output vector and keeping track in matrix
        for index in range(x_train_subset.shape[1]):
            # pulling the ith input vector from our x_train subset matrix.

            # Note: we transpose because column slice auto reverts to a row vector
            curr_input_vec = np.array([x_train_subset[:, index]]).T

            # pulling the model output w.r.t the ith input vector
            curr_output_vec = push_output(curr_input_vec)

            # appending our model output prediction to our output vector
            output_vec[index, :] = curr_output_vec.T #check why this is transposed b4 being added again

        # returning our matrix of model predictions
        return output_vec

    def train(self):
        """
        This method trains our model to the x/y training data.
        """
        # init the index counter for our error log
        error_log_index = -1
        for epoch in range(self.epoch_num):
            # shuffling training data
            np.random.shuffle(self.train_data)
            if self.train_data.shape[0] % self.batch_size != 0:
                # checking to make sure batches can be split even;y
                raise Exception('Batch_size must evenly divide training data!')
            else:
                for sub_data in np.split(self.train_data, self.batch_size):
                    # incrementing error_log
                    error_log_index += 1
                    # splitting sub-data into x input vector and associated y output  vector
                    x_train_subset, y_train_subset = sub_data[:, :self.train_split_index], sub_data[:, self.train_split_index:]

                    # pulling our model's prediction matrix on the x_train_subset data
                    prediction_matrix = np.array([self.forward_propagation(x_train_subset.T)])

                    # printing model error every prediction batch
                    if self.cost_func == 'mse':
                        self.print_error = self.mean_squared_error(prediction_matrix, y_train_subset)
                    elif self.cost_func == 'log-loss':
                        self.print_error = self.log_loss_error(prediction_matrix, y_train_subset)

                    # adding current model error to log
                    self.error_log[error_log_index] = self.print_error

                    # updating params via gradient descent + backpropagation
                    self.backpropagation(x_train_subset.T, prediction_matrix, y_train_subset)

            # printing epoch number
            print(f'Epoch # {epoch + 1}')

    def backpropagation(self, x_inputs_vector, y_predictions_vector, y_observation_vector):

        #init our overall error prime for the n observations in the x_inputs_vector
        error_prime_vec = self.mse_prime(y_predictions_vector, y_observation_vector,
                                         output_layer_depth=self.network[-1].neuron_number)

        #MAKE SURE EACH x_input_vec is  column vector

        for col_index in range(x_inputs_vector.shape[1]):
            x_input_vec = np.atleast_2d(x_inputs_vector[:,col_index]).T

            # iterating backwards through the self.network Layer np array
            for index in range(len(self.network) - 1, -1, -1):

                # pulling layers in reverse
                curr_layer = self.network[index]
                if curr_layer.layer_type == 'output':

                    #pulling previous layer activation
                    prev_layer_activation = self.network[index - 1].output_vector

                    #calculating weights partial (error is logged internally)
                    weight_partial = curr_layer.calculate_output_layer_partial(error_prime_vec, prev_layer_activation)

                    #updating weights/biase
                    curr_layer.update_biases(self.learn_rate)
                    curr_layer.update_weights(self.learn_rate,weight_partial)

                else:

                    #pulling next layer weights/error vector
                    next_layer_weights = self.network[index + 1].old_weights
                    next_layer_error_vec = self.network[index + 1].error_vec

                    #pulling previous layer's activation
                    if curr_layer.layer_type == 'hidden':
                        prev_layer_activation = self.network[index - 1].output_vector
                    elif  curr_layer.layer_type == 'initial_hidden':
                        #BACKPROP OVER EACH ENTRY IN X_INPUTS ARR
                        prev_layer_activation = x_input_vec #MAKE SURE THIS IS nX1

                    weight_partial = curr_layer.calculate_hidden_layer_partial(next_layer_weights,next_layer_error_vec,prev_layer_activation)

                    # updating weights/biase
                    curr_layer.update_biases(self.learn_rate)
                    curr_layer.update_weights(self.learn_rate, weight_partial)

    def mean_squared_error(self, y_predictions_vector, y_observation_vector):
        """
        This function returns the means squared error associated with our prediction vector
        Used for printing how our model is fitting to the data each epoc (regression)
        """
        return 1 / self.batch_size * np.sum((y_observation_vector - y_predictions_vector) ** 2)

    def mse_prime(self, y_predictions_vector, y_observation_vector, output_layer_depth):
        """
        This function returns the derivative of mse w.r.t the y_predictions_vector. Used in backpropagation
        We stack our average errors w.r.t the number of neurons in our output layer
        """
        # note: may have to alter the # of columns returned from this func if we have 2+ nodes in output layer

        return np.array(
            [[-2 / self.batch_size * np.sum((y_observation_vector - y_predictions_vector))] * self.y_test.shape[1]
             for _ in range(output_layer_depth)])

    def log_loss_error(self, y_predictions_vector, y_observation_vector):
        """
        This function returns the log loss error associated with our prediction vector
        Used for printing how our model is fitting to the data each epoc (classification)
        """
        return -1 / self.batch_size * np.sum(
            y_observation_vector * np.log(y_predictions_vector) + (1 - y_observation_vector) * np.log(
                1 - y_predictions_vector))

    def log_loss_prime(self, y_predictions_vector, y_observation_vector, output_layer_depth):
        """
        This function returns the derivative of log loss w.r.t the y_predictions_vector. Used in backpropagation
        """
        return np.array([[1 / self.batch_size * np.sum(
            -(y_observation_vector / y_predictions_vector) + (1 - y_observation_vector) / (1 - y_predictions_vector))] *
                         self.y_test.shape[1]
                         for _ in range(output_layer_depth)])

    def plot_train_error(self):
        """
        this method plots our error while training
        """
        plt.plot(np.arange(1, len(self.error_log) + 1), self.error_log)
        plt.title('Model Error While')
        plt.xlabel('Training Batch')
        plt.ylabel('Model Error')
        plt.show()
        # printing error log to terminal
        print(self.error_log)

    def test_regression(self):
        """
        this method tests the regression capabilities of our model
        """
        model_predictions = self.forward_propagation(self.x_test.T)
        # calculating & displaying MSE/RMSE
        mse = self.mean_squared_error(model_predictions, self.y_test)
        rmse = (self.mean_squared_error(model_predictions, self.y_test)) ** 0.5
        print(f'Regression Model has an MSE accuracy of: {mse}')
        print(f'Regression Model has an RMSE accuracy of: {rmse}')

    def test_classification(self):
        """
        this method tests the classification capabilities of our model
        """
        model_predictions = self.forward_propagation(self.x_test.T)
        right_predicts = 0
        for prediction, observation in zip(model_predictions, self.y_test):
            if observation == 1 and 0.5 <= prediction <= 1:
                right_predicts += 1
            elif observation == 0 and 0 <= prediction < 0.5:
                right_predicts += 1
            else:
                pass
        model_accuracy = right_predicts / model_predictions.shape[0] * 100
        print(f'The Binary Classification Model has an accuracy of {model_accuracy}%')

    def __repr__(self):
        return f'Network trained with {self.cost_func} Cost Function. Layers: \n' \
               f'{self.network}'


if __name__ == '__main__':
    # testing the forward propagation of a network which solves the Binary And problem

    # setting up test/train data arrays #CLEAN UP AND CHECK FOR ACCURACY
    X_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    Y_train = np.array([[0], [1], [1], [0]])
    Y_train_2 = np.array([[0], [0], [0], [1]])
    X_test = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1,1], [1, 0], [1, 1]])
    Y_test = np.array([[0], [0], [1], [1], [0], [1], [1], [0]])
    Y_test_2 = np.array([[0], [0], [0], [1], [1], [1], [0],[1]])

    # creating model object
    model = Network('mse', X_train, Y_train, X_train, Y_train, x_feautures=2,epoch_num=500, batch_size=1,
                    layer_num=2, layer_depths=[2, 1], learn_rate=0.03)

    # adding hidden layer
    model.add_Layer('relu', 'initial_hidden')

    # adding output layer
    model.add_Layer('sigmoid', 'output')

    # printing the model
    print(model)

    # training model
    model.train()


    # plotting model error while training
    model.plot_train_error()

    print('Optimized Model: ')
    print(model)

    # testing model
    model.test_classification()
