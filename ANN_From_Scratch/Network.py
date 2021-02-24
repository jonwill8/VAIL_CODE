# Future Goal 1 --> Implement Parameter Regularization Techniques (ex: Neuron Dropout)
# Future Goal 2 --> Implement Softmax for Multinomial Classification

# making necessary imports
import numpy as np
import matplotlib.pyplot as plt
from Layer import Layer


class Network:
    """
    This Class models a Fully Connected ANN.
    The NN can perform both Binary classification, Multinomial Classification, & Regression

    DATA ENTRY NOTE:
    all x/y train & test data must be passed as 2D np arrays!
    Each row holds the complete x/y data for a single test or train sample
    """

    def __init__(self, cost_func, x_train, y_train, x_test, y_test, x_features, epoch_num, layer_num,
                 layer_depths, batch_num, learn_rate):
        """
        Constructor to Initialize appropriate Model Instance Variables
        """
        if cost_func.lower() not in ('mse', 'log-loss'):
            raise Exception('Must Provide a Valid Model Cost Function')
        else:
            self.cost_func = cost_func.lower()

        # check to ensure all data is in 2d np arrays
        if x_train.ndim == 1 or y_train.ndim == 1 or x_test.ndim == 1 or y_test.ndim == 1:
            raise Exception('Passed Train & Test Data must be wrapped in a 2D numpy array!')
        if len(x_train.shape) != 2 or len(y_train.shape) != 2 or len(x_test.shape) != 2 or len(y_test.shape) != 2:
            raise Exception('Passed Train & Test Data must be wrapped in a 2D numpy array!'
                            'Each row corresponds to the data for a single test/train sample')

        # creating np array which holds both x_train & associated y train
        self.train_split_index = x_train.shape[1]  # creating the split index which separates x/y data
        self.train_data = np.append(x_train, y_train, axis=1)  # creating our macro training data np array
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

        # checking to see if batch_num can evenly divide our training data np array
        if self.train_data.shape[0] % batch_num != 0:
            raise Exception('You must choose a batch number which evenly divides the training data!')
        else:
            self.batch_num = batch_num
        self.learn_rate = learn_rate
        self.error_log = np.zeros((self.epoch_num * self.batch_num))

    def add_Layer(self, neuron_activation_func, layer_type):
        """
        Adds a Layer object to our self.network np array
        Note: when adding layers the order must be: initial_hidden --> hidden_1 --> hidden_2 --> hidden_n --> output
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
        layer = Layer(neuron_number, neuron_activation_func, layer_type, fan_in, fan_out)
        self.network[add_index] = layer

    def train(self):
        """
        This method trains our model to the x/y training data.
        """
        # init the index counter for our error log
        error_log_index = -1
        for epoch in range(self.epoch_num):
            # shuffling training data
            np.random.shuffle(self.train_data)
            # splitting training data by batch size
            for train_data_subset in np.split(self.train_data, self.batch_num):
                # incrementing the error log index
                error_log_index += 1
                # pulling the x & y data vectors from our train_batch_subset
                x_train_batch = train_data_subset[:, :self.train_split_index]
                y_train_batch = train_data_subset[:, self.train_split_index:]
                # pulling our model predictions on the x_data vectors from our train_batch_subset
                prediction_matrix = self.predict(x_train_batch)
                # calculating batch error
                if self.cost_func == 'mse':
                    batch_error = self.mean_squared_error(prediction_matrix, y_train_batch)
                elif self.cost_func == 'log-loss':
                    batch_error = self.log_loss_error(prediction_matrix, y_train_batch)
                # adding current model error to log
                self.error_log[error_log_index] = batch_error
                # updating params via gradient descent + backpropagation
                self.backpropagation(x_train_batch, prediction_matrix, y_train_batch)

            # printing error after each epoch nth epoch number
            print(f'Epoch #: {epoch + 1} | Latest Error" {self.error_log[error_log_index]}')

    def predict(self, x_input_vec):
        """
        this method yields a model prediction array for the provided x_input_vec
        pass x_inputs in the form:
        [[x inputs 1]
         [x inputs 2]
         [x inputs n]]
        """
        # catching bad input data dimensions
        if x_input_vec.ndim == 1 or len(x_input_vec.shape) != 2:
            raise Exception('X_inputs vector must be 2-D np array. '
                            'Each row corresponds to the x variable observations for a single sample!')

        # creating function to return model predictions for given x_input_vector in column form
        def push_output(input_vector):
            """
            this function passes the initial input vector through the entire network.
            Returns the prediction vector from our network's output layer
            """
            output = input_vector
            for layer in self.network:
                output = layer.gen_output_vector(output)
            return output

        # creating output vector
        outputs_vec = np.zeros((x_input_vec.shape[0], self.y_test.shape[1]))

        # iterating over all x input rows in the x_input_vec
        for row_index in range(x_input_vec.shape[0]):
            curr_x_input_vec = np.atleast_2d(x_input_vec[row_index, :])
            # transposing so we can feed into the model
            curr_x_input_vec = curr_x_input_vec.T
            # feeding our x input vector into the model
            curr_output = push_output(curr_x_input_vec)
            outputs_vec[row_index, :] = curr_output.T

        # returning our output vector
        return outputs_vec

    def backpropagation(self, x_inputs_vector, y_predictions_vector, y_observation_vector):
        # init our overall error prime vector

        if self.cost_func == 'mse':
            error_prime_vec = self.mse_prime(y_predictions_vector, y_observation_vector)
        elif self.cost_func == 'log-loss':
            error_prime_vec = self.log_loss_prime(y_predictions_vector, y_observation_vector)

        for row_index in range(x_inputs_vector.shape[0]):

            # making sure each x vec is a column vec
            x_input_vec = np.atleast_2d(x_inputs_vector[row_index, :]).T

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
                        prev_layer_activation = x_input_vec

                    # calculating weights partial (error is logged internally)
                    weight_partial = curr_layer.calculate_hidden_layer_partial(next_layer_weights, next_layer_error_vec,
                                                                               prev_layer_activation)

                    # updating weights/bias
                    curr_layer.update_biases(self.learn_rate)
                    curr_layer.update_weights(self.learn_rate, weight_partial)

    @staticmethod
    def mean_squared_error(y_predictions_vector, y_observation_vector):
        """
        This function returns the mean squared error associated with our model's prediction
        Used for printing how our model is fitting to the data each epoc (regression)
        """
        return np.mean(np.sum((y_observation_vector - y_predictions_vector) ** 2))

    @staticmethod
    def mse_prime(y_predictions_vector, y_observation_vector):
        """
        Returns the single sample derivative of mse w.r.t the y_predictions_vector. Used in backpropagation
        Return 2d vector dimensions are n x 1 (column vector)
        """

        error_gradients = (np.mean((-2*(y_observation_vector - y_predictions_vector)), axis=0)).T
        return error_gradients.reshape(error_gradients.shape[0], -1)

    @staticmethod
    def log_loss_error(y_predictions_vector, y_observation_vector):
        """
        This function returns the log loss error associated with our prediction vector
        Used for printing how our model is fitting to the data each epoc (classification)
        """
        return -1 * np.mean(y_observation_vector * np.log(y_predictions_vector) + (1 - y_observation_vector) * np.log(
            1 - y_predictions_vector))

    @staticmethod
    def log_loss_prime(y_predictions_vector, y_observation_vector):
        """
        This function returns the derivative of log loss w.r.t the y_predictions_vector. Used in backpropagation
        """
        error_gradients = (np.mean(((-y_observation_vector / y_predictions_vector) + (1 - y_observation_vector) / (1 - y_predictions_vector)), axis=0)).T
        return error_gradients.reshape(error_gradients.shape[0], -1)

    def plot_train_error(self):
        """
        this method plots our model error over training batches
        """
        plt.plot(np.arange(1, len(self.error_log) + 1), self.error_log)
        plt.title('Model Error Over Training Batches')
        plt.xlabel('Batch #')
        plt.ylabel('Model Error')
        plt.show()

    def test_regression(self):
        """
        this method tests the regression capabilities of our model
        """
        model_predictions = self.predict(self.x_test)
        # calculating & displaying MSE/RMSE
        mse = float(self.mean_squared_error(model_predictions, self.y_test))
        rmse = mse ** 0.5
        print(f'Regression Model has an MSE accuracy of: {round(mse, 4)}')
        print(f'Regression Model has an RMSE accuracy of: {round(rmse, 4)}')

    def test_binary_classification(self):
        """
        this method tests the binary classification accuracy of our model
        """
        model_predictions = self.predict(self.x_test)
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

    def test_multinomial_classification(self):
        """
        This method tests the multinomial classification accuracy of our model
        """
        model_predictions = self.predict(self.x_test)
        right_predicts = 0

        for row_index in range(self.y_test.shape[0]):
            curr_y_vec = self.y_test[row_index, :]
            curr_model_predict = model_predictions[row_index, :]
            if np.argmax(curr_model_predict) == np.argmax(curr_y_vec):
                right_predicts += 1

        model_accuracy = right_predicts / model_predictions.shape[0] * 100
        print(f'The Multinomial Classification Model has an accuracy of {round(model_accuracy, 4)}%')

    def __repr__(self):
        return f'Network trained with {self.cost_func} Cost Function. Layers: \n' \
               f'{self.network}'
