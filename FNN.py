import numpy as np
class FeedForwardNetwork:

    def __init__(self, number_of_inputs, hidden_layers_unit, number_of_outputs):
        self.number_of_inputs = number_of_inputs
        self.hidden_layers_unit = hidden_layers_unit
        self.number_of_outputs = number_of_outputs
        self.layers = None

        #If first row for hidden layer has no units then there are no hidden layers
        if self.hidden_layers_unit[0] == 0:
            self.layers = [self.number_of_inputs] + [self.number_of_outputs]

        else:
            self.layers = [self.number_of_inputs] + self.hidden_layers_unit + [self.number_of_outputs]

        self.weights = []
        self.bias = []
        self.previous_weights = []
        self.previous_bias = []
        self.activation_values = []
        self.gradient_descent_values = []

        #i+1h+o = 2 weights = i + #h

        for unit_layer in range(len(self.layers) - 1):

            #create 2D array of randomized weights
            #num rows = num of neurons in layer unit_layer
            # num columns = num of neurons in layer unit_layer + 1

            #90% of weights betweeen -2 and 2 but since this is minimal -1 and 1 is prob better
            the_weights = 2 * np.random.rand(self.layers[unit_layer], self.layers[unit_layer + 1]) - 1
            the_bias = 2 * np.random.rand(self.layers[unit_layer + 1]) - 1

            self.weights.append(the_weights)

            self.bias.append(the_bias)

            self.previous_weights.append(the_weights)

            self.previous_bias.append(the_bias)

        for unit_layer in range(len(self.layers)):
            activation = np.zeros(self.layers[unit_layer])
            self.activation_values.append(activation)

        for unit_layer in range(len(self.layers) - 1):
            updated_values = np.zeros(self.layers[unit_layer])
            self.gradient_descent_values.append(updated_values)

    """
    Sigmoid activation function
    """
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    """
    Gradient of the sigmoid i.e loss function
    """
    def sigmoid_derivative(self, x):
        return x * (1 - x)


    """
    error measure 
    """
    def mean_squared_error(self, predicted, actual):
        return np.average((predicted - actual) ** 2)

    def forward_propagation(self, inputs):

        self.activation_values[0] = inputs
        activations = inputs

        for unit_layer, weights in enumerate(self.weights):

            linear_function = np.dot(activations, weights) + self.bias[unit_layer]

            self.activation_values[unit_layer + 1] = self.sigmoid(linear_function)
            activations = self.sigmoid(linear_function)

        return self.activation_values[unit_layer + 1]

    def backward_propagation(self, error_rate):
        # iterate starting from output and stop before input layer
        for unit_layer in reversed(range(len(self.activation_values) - 1)):

            # output layers
            if (unit_layer + 1) == len(self.activation_values) - 1:
                self.gradient_descent_values[unit_layer] = error_rate * self.sigmoid_derivative(
                    self.activation_values[unit_layer + 1])

            # hidden layers
            else:
                self.gradient_descent_values[unit_layer] = np.dot(self.gradient_descent_values[unit_layer + 1],  self.weights[unit_layer + 1].transpose()) * self.sigmoid_derivative(
                    self.activation_values[unit_layer + 1])

    def gradient_descent(self, learning_rate = 0.05):
       for unit_layer in range(len(self.weights)):
           self.weights[unit_layer] = self.weights[unit_layer] + learning_rate * np.outer(self.activation_values[unit_layer], self.gradient_descent_values[unit_layer])

           self.bias[unit_layer] = self.bias[unit_layer] + learning_rate * self.gradient_descent_values[unit_layer]

           self.previous_weights[unit_layer] = self.weights[unit_layer]

           self.previous_bias[unit_layer] = self.bias[unit_layer]

    def train(self, training_inputs, training_outputs, learning_rate):
        total_error = 0
        success = 0

        for i, input in enumerate(training_inputs):

            actual = training_outputs[i]

            predicted = self.forward_propagation(input)

            error_rate = actual - predicted
            if error_rate < 0.5:
                success += 1

            self.backward_propagation(error_rate)

            self.gradient_descent(learning_rate)

            total_error += self.mean_squared_error(predicted, actual)

        average_error = total_error / len(training_inputs)

        success_rate = success / len(training_inputs)

        return average_error, success_rate

    def test(self, test_inputs, test_outputs):
        total_error = 0
        success = 0

        for i, input in enumerate(test_inputs):
            actual = test_outputs[i]

            predicted = self.forward_propagation(input)

            error_rate = actual - predicted

            if error_rate < 0.5:
                success += 1

            total_error += self.mean_squared_error(actual, predicted)

        average_error = total_error / len(testing_input)

        success_rate = success / len(testing_input)

        return average_error, success_rate

network = FeedForwardNetwork(4, [4, 4], 1)

training_input = np.array([[0,0,1,1],[0,1,0,1]], dtype=float)

training_output = np.array([0,1,1,0], dtype=float)

error_rate, success_rate = network.train(training_inputs = training_input, training_outputs = training_output, learning_rate=0.9)
print(error_rate, success_rate)

testing_input = np.array([[1,0,0,1,],[0,1,0,1]], dtype=float)

testing_output = np.array([1,1,0,0], dtype=float)

error_rate, success_rate = network.test(test_inputs = testing_input, test_outputs = testing_output)
print(error_rate, success_rate )
