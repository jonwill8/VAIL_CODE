"""
This script contains the Macro Network Class for our fully connected ANN
This ANN will perform both Binary classification (log loss GD optimization) and regression (MSE loss GD optimization)

DATA ENTRY NOTE:

all x/y train & test data must be passed as 2D Np arrays
each row corresponds to the x input vector of a single y train/test samples
data must always be of form a x b where a is the number of samples and b is the number of features
"""

# TODO Implement Parameter Regularization Techniques (ex: Neuron Dropout)
# TODO Implement Mini Batch GD
# TODO Implement Classification Grading multinomial


# making necessary imports
from Layer_V2 import Layer2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils


class Network:

    def __init__(self, cost_func, x_train, y_train, x_test, y_test, x_features, epoch_num, layer_num,
                 layer_depths,learn_rate,print_error_iteration):
        # init appropriate IVs
        if cost_func.lower() not in ('mse', 'log-loss'):
            raise Exception('Must Provide a Valid Model Cost Function')
        else:
            self.cost_func = cost_func.lower()

        # check to ensure all data is in 2d vectors
        if x_train.ndim == 1 or y_train.ndim == 1 or x_test.ndim == 1 or y_test.ndim == 1:
            raise Exception('Passed Train & Test Data must be wrapped in a 2D numpy array!')
        if len(x_train.shape) != 2 or len(y_train.shape) != 2 or len(x_test.shape) != 2 or len(y_test.shape) !=2:
            raise Exception('Passed Train & Test Data must be wrapped in a 2D numpy array!'
                            'Each row corresponds to the data for a single test/train sample')

        # creating np array which holds both x_train & associated y train
        # note: you split the train_data np array at the self.split index
        # x_arr = self.train_data[:,:self.train_split_index]
        # y_arr = self.train_data[:,train_split_index:]
        self.train_split_index = x_train.shape[1]
        self.train_data = np.append(x_train, y_train, axis=1)
        self.x_test = x_test
        self.y_test = y_test

        # init the number of x features we have each sample
        self.x_features = x_features

        # creating our np array of layer objects
        self.network = np.empty(layer_num, dtype=object)

        # init all other IVs
        self.epoch_num = epoch_num
        self.layer_num = layer_num
        self.layer_depths = layer_depths
        self.learn_rate = learn_rate
        # we log an error each batch during the SGD
        self.batch_size = self.train_data.shape[0]
        self.error_log = np.zeros((self.epoch_num *  self.batch_size))
        self.print_error = None
        self.print_error_iteration = print_error_iteration

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
            fan_in = self.x_features  # should match the # of x variables that are tied to each y output
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
        This method feeds x input vectors through the model & appends our predictions to an np array (self.predictions)
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
            output_vec[index, :] = curr_output_vec.T  # check why this is transposed b4 being added again

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
            # spltting training data into single sample vectors
            for sub_data in np.split(self.train_data, self.batch_size): #single sample SGD
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

        # printing every nth epoch number
        if (epoch + 1) % self.print_error_iteration == 0:
            print(f'Epoch # {epoch + 1}')

    def backpropagation(self, x_inputs_vector, y_predictions_vector, y_observation_vector):

        # init our overall error prime for the n observations in the x_inputs_vector

        if self.cost_func == 'mse':
            error_prime_vec = self.mse_prime(y_predictions_vector, y_observation_vector,
                                         output_layer_depth=self.network[-1].neuron_number)
        elif self.cost_func == 'log-loss':
            error_prime_vec = self.log_loss_prime(y_predictions_vector, y_observation_vector,
                                             output_layer_depth=self.network[-1].neuron_number)

        for col_index in range(x_inputs_vector.shape[1]):
            x_input_vec = np.atleast_2d(x_inputs_vector[:, col_index]).T #making sure each x input vec is a column vec

            # iterating backwards through the self.network Layer np array
            for index in range(len(self.network) - 1, -1, -1):

                # pulling layers in reverse
                curr_layer = self.network[index]
                if curr_layer.layer_type == 'output':

                    # pulling previous layer activation
                    prev_layer_activation = self.network[index - 1].output_vector

                    # calculating weights partial (error is logged internally)
                    weight_partial = curr_layer.calculate_output_layer_partial(error_prime_vec, prev_layer_activation)

                    # updating weights/bias
                    curr_layer.update_biases(self.learn_rate)
                    curr_layer.update_weights(self.learn_rate, weight_partial)

                else:

                    # pulling next layer weights/error vector
                    next_layer_weights = self.network[index + 1].old_weights
                    next_layer_error_vec = self.network[index + 1].error_vec

                    # pulling previous layer's activation
                    if curr_layer.layer_type == 'hidden':
                        prev_layer_activation = self.network[index - 1].output_vector
                    elif curr_layer.layer_type == 'initial_hidden':
                        # backpropagate OVER EACH ENTRY IN X_INPUTS ARR
                        prev_layer_activation = x_input_vec  # MAKE SURE THIS IS nX1

                    weight_partial = curr_layer.calculate_hidden_layer_partial(next_layer_weights, next_layer_error_vec,
                                                                               prev_layer_activation)

                    # updating weights/bias
                    curr_layer.update_biases(self.learn_rate)
                    curr_layer.update_weights(self.learn_rate, weight_partial)

    def predict(self, x_input_vec):
        """
        this method yields a single model prediction for the provided x_input_vec
        pass x_inputs in the form:

        [x inputs 1]
        [x inputs 2]
        [x inputs n]
        """
        #catching bad input data dimensions
        if x_input_vec.ndim == 1:
            raise Exception('Passed x_inputs vector must be 2-D!')

        #creating our function to return model predictions for given x_input_vector in column form
        def push_output(input_vector):
            # this function passes the initial input vector through the entire network.
            # Returns the prediction vector from our network's output layer
            output = input_vector
            for layer in self.network:
                output = layer.gen_output_vector(output)
            return output
        #creating output vector
        outputs_vec = np.zeros((x_input_vec.shape[0],1))
        #iterating over all x input rows in the x_input_vec
        for row_index in range(x_input_vec.shape[0]):
            curr_x_input_vec = np.atleast_2d(x_input_vec[row_index,:])
            #transposing so we can feed into the model
            curr_x_input_vec = curr_x_input_vec.T
            #feeding our x input vector into the model
            curr_output = push_output(curr_x_input_vec)
            outputs_vec[row_index,:] = curr_output
        #printing our output vector
        print(outputs_vec)

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
        """
        THIS CODE WORKED FOR XOR CLASSIFICATION!
        return np.array(
            [[-2 / self.batch_size * np.sum((y_observation_vector - y_predictions_vector))] * self.y_test.shape[1]
             for _ in range(output_layer_depth)])
        """
        return np.array(
            [[-2 / self.batch_size * np.sum((y_observation_vector - y_predictions_vector))]
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

    def plot_train_error(self,print_error=False):
        """
        this method plots our error while training
        """
        plt.plot(np.arange(1, len(self.error_log) + 1), self.error_log)
        plt.title('Model Error Over Training Iterations')
        plt.xlabel('Batch #')
        plt.ylabel('Model Error')
        plt.show()
        # printing every 50th error from error log to terminal
        if print_error == True:
            for index, error_val in enumerate(self.error_log):
                if (index + 1) % 50 == 0:
                    print(error_val)

    def test_regression(self):
        """
        this method tests the regression capabilities of our model
        """
        model_predictions = self.forward_propagation(self.x_test.T)
        # calculating & displaying MSE/RMSE
        mse = self.mean_squared_error(model_predictions, self.y_test)
        rmse = mse ** 0.5
        print(f'Regression Model has an MSE accuracy of: {round(mse, 4)}')
        print(f'Regression Model has an RMSE accuracy of: {round(rmse, 4)}')

    def test_classification(self):
        """
        this method tests the classification capabilities of our model
        """
        model_predictions = self.forward_propagation(self.x_test.T)
        right_predicts = 0
        for prediction, observation in zip(model_predictions, self.y_test):
            if observation == 1 and 0.5 <= prediction <= 1:
                right_predicts += 1
            elif observation == 0 and prediction < 0.5:
                right_predicts += 1
            else:
                pass
        model_accuracy = right_predicts / model_predictions.shape[0] * 100
        print(f'The Binary Classification Model has an accuracy of {round(model_accuracy, 4)}%')

    def __repr__(self):
        return f'Network trained with {self.cost_func} Cost Function. Layers: \n' \
               f'{self.network}'


if __name__ == '__main__':

    # load MNIST from server
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # %%

    # reshaping & normalizing x_train data
    x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
    x_train = x_train.astype('float32')
    x_train /= 255

    # hot encoding y_train vector
    y_train = np_utils.to_categorical(y_train)

    # reshaping & normalizing y_train data
    x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = np_utils.to_categorical(y_test)

    #taking 2000 subsamples:
    x_train_trunc = x_train[0:2000]
    y_train_trunc = y_train[0:2000]
    
    #testing on 50 samples
    x_test_trunc = x_test[0:50]
    y_test_trunc = y_test[0:50]

    #reshaping data to conform to our model 2d np arrray standards
    x_train_trunc = np.reshape(x_train_trunc, (x_train_trunc.shape[0], x_train_trunc.shape[2]))
    #y_train_trunc = np.reshape(y_train_trunc, (y_train_trunc.shape[0], y_train_trunc.shape[2]))
    x_test_trunc = np.reshape(x_test_trunc, (x_test_trunc.shape[0], x_test_trunc.shape[2]))
    #y_test_trunc =  np.reshape(y_test_trunc, (y_test_trunc.shape[0], y_test_trunc.shape[2]))


    # creating our model
    model = Network('mse', x_train_trunc, y_train_trunc, x_test_trunc, y_test_trunc, x_features=784, epoch_num=35,
                    layer_num=3, layer_depths=[100, 50, 10], learn_rate=0.1,print_error_iteration=1)

    #adding our tangent hyperbolic hidden layers
    model.add_Layer('tanhyp', 'initial_hidden')
    model.add_Layer('tanhyp', 'hidden')

    # adding output layer
    model.add_Layer('tanhyp', 'output')

    #training model
    model.train()

    #plotting model error while training
    model.plot_train_error(print_error=False)




