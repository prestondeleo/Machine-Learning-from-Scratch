import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

dataStuffs = pd.read_csv("irisdata.csv")

dataStuffs = dataStuffs.drop(dataStuffs[dataStuffs.species == 'setosa'].index)

dataStuffs = dataStuffs.copy()

dataStuffs.reset_index(inplace=True)

some_data = dataStuffs.iloc[:, :-1]

X_to_use = dataStuffs.iloc[:, 3:5]

the_data = X_to_use.values.tolist()

species = dataStuffs.iloc[:, -1]

np.random.seed(47)

class One_layer_classifier:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def value_of_sigmoid(self, x_first, x_second):
        delta_y = self.weights[0] * x_first + self.weights[1] * x_second + self.bias
        classification = self.sigmoid(delta_y)
        return classification

    def non_lin_function(self):
        slope = -(self.weights[0] / self.weights[1])

        intercept = -(self.bias / self.weights[1])

        x_values = np.linspace(3, 7, num=100)

        y_values = slope * x_values + intercept

        return x_values, y_values


class NeuralNetworks:

    def __init__(self, data_vectors, some_weights, the_bias, pattern_classes):

        self.data_vectors = data_vectors
        self.pattern_classes = pattern_classes
        self.some_weights = some_weights
        self.the_bias = the_bias

    def get_bias(self):
        return self.the_bias

    def get_weights(self):
        return self.some_weights

    def get_first_weight(self):
        return self.some_weights[0]

    def get_second_weight(self):
        return self.some_weights[1]

    def set_bias(self, altered_bias):
        self.the_bias = altered_bias

    def set_weights(self, altered_weights):
        self.some_weights = altered_weights

    def set_first_weight(self, new_first_weight):
        self.some_weights[0] = new_first_weight

    def set_second_weight(self, new_second_weight):
        self.some_weights[1] = new_second_weight

    def non_lin_function(self):

        slope = -(self.some_weights[0] / self.some_weights[1])

        intercept = -(self.the_bias / self.some_weights[1])

        x_values = np.linspace(3, 7, num=100)

        y_values = slope * x_values + intercept

        return x_values, y_values

    def mse(self):

        classifier = One_layer_classifier(self.some_weights, self.the_bias)

        summation = 0

        for vector in range(len(self.data_vectors)):

            y_knot = 0

            if (self.pattern_classes[vector] == 'virginica'):
                y_knot = 1

            elif (self.pattern_classes[vector] == 'versicolor'):
                y_knot = 0

            x_first = self.data_vectors[vector][0]

            x_second = self.data_vectors[vector][1]

            y_hat = classifier.value_of_sigmoid(x_first, x_second)

            summation += (y_knot - y_hat) ** 2

        mse = summation / len(self.data_vectors)

        return mse

    def summed_gradient(self, weights, bias):
        classifier = One_layer_classifier(weights, bias)

        gradient_sum = [0, 0, 0]
        y_knot = 0

        for vector in range(len(self.data_vectors)):

            if (self.pattern_classes[vector] == 'virginica'):
                y_knot = 1

            elif (self.pattern_classes[vector] == 'versicolor'):
                y_knot = 0

            x_first = self.data_vectors[vector][0]

            x_second = self.data_vectors[vector][1]

            value_of_sig = classifier.value_of_sigmoid(x_first, x_second)

            d_1 = (-2 * (y_knot - value_of_sig) * value_of_sig * (1 - value_of_sig) * self.data_vectors[vector][0]) / len(self.data_vectors)

            d_2 = (-2 * (y_knot - value_of_sig) * value_of_sig * (1 - value_of_sig) * self.data_vectors[vector][1]) / len(self.data_vectors)

            # W__0 is constant so ommitted x value in calcultaion
            d_0 = (-2 * (y_knot - value_of_sig) * value_of_sig * (1 - value_of_sig)) / len(self.data_vectors)

            gradient_sum[0] += d_0
            gradient_sum[1] += d_1
            gradient_sum[2] += d_2

        return gradient_sum

    def gradient_descent(self, epsilon, stopping_criterion):

        mse_list = []
        weights_list = []
        bias_list = []

        original_bias = self.get_bias()

        original_weights = self.get_weights().copy()

        gradient = self.summed_gradient(self.get_weights(), self.get_bias())

        i = 1
        while (self.mse() > stopping_criterion):
            mse_list.append(self.mse())

            weights_list.append(self.get_weights().copy())

            bias_list.append(self.get_bias())

            #updating bias = bias - epsilon * gradient_bias
            self.set_bias(self.get_bias() - epsilon * gradient[0])

            # updated new_first_weight = first_weight - epsilon * gradient_weight_1
            # new_first_weight =
            self.set_first_weight(self.get_first_weight() - epsilon * gradient[1])

            # updated new_second_weight = second_weight - epsilon * gradient_weight_2
            # new_second_weight =
            self.set_second_weight(self.get_second_weight() - epsilon * gradient[2])

            gradient = self.summed_gradient(self.get_weights(), self.get_bias())

            i += 1
        #Resetting bias and weights if another round occurs
        self.set_bias(original_bias)
        self.set_weights(original_weights.copy())

        return bias_list, weights_list, mse_list, i

    #if number of plots is 3 shows first, middle, and last. All shows all decision boundaries
    def plot_decision_boundaries(self, bias_list, weights_list, number_of_plots, iterations):
        fig, ax = plt.subplots()
        ax.set_xlabel('petal length')
        ax.set_ylabel('petal width')

        colors = {'versicolor': 'g', 'virginica': 'b'}
        markers = {'versicolor': '$*$', 'virginica': '+'}

        # fig, ax = plt.subplots()

        for i in range(len(some_data['petal_length'])):
            ax.scatter(some_data['petal_length'][i], some_data['petal_width'][i], color=colors[species[i]],
                       marker=markers[species[i]])

        if (number_of_plots == 'all'):
            for element in range(len(bias_list)):
                self.set_bias(bias_list[element])
                self.set_weights(weights_list[element])

                x_values, y_values = self.non_lin_function()

                plt.plot(x_values, y_values)

        elif (number_of_plots == '3'):
            self.set_bias(bias_list[0])
            self.set_weights(weights_list[0])
            x_values, y_values = self.non_lin_function()
            plt.plot(x_values, y_values, label='initial line')
            print(bias_list[0])
            print(weights_list[0])

            self.set_bias(bias_list[round(iterations / 2)])
            self.set_weights(weights_list[round(iterations / 2)])
            x_values, y_values = self.non_lin_function()
            plt.plot(x_values, y_values, label='middle line')

            self.set_bias(bias_list[iterations - 2])
            self.set_weights(weights_list[iterations - 2])
            x_values, y_values = self.non_lin_function()
            plt.plot(x_values, y_values, label='final line')

            plt.legend()

        plt.show()
        # plt.close()

    def plot_learning_curve(self, list_of_mse_values, iterations):
        fig, ax = plt.subplots()

        ax.set_xlabel('iterations')
        ax.set_ylabel('MSE')
        plt.plot(range(iterations - 1), list_of_mse_values)
        plt.show()
        # plt.close()

    def randomized_weights(self):
        og_weights = self.get_weights().copy()
        og_bias = self.get_bias()
        np.random.seed(47)
        weight_one = np.random.uniform(1, 4)
        np.random.seed(48)
        weight_two = np.random.uniform(-3, -1)
        np.random.seed(47)
        random_bias = np.random.uniform(-3, 3)

        random_weights = [weight_one, weight_two]
        print(random_weights)
        print(random_bias)

        self.set_weights(random_weights.copy())
        self.set_bias(random_bias)

        new_bias_list1, new_weight_list1, mse_value1, iterations1 = self.gradient_descent(0.1, 0.05)

        self.plot_decision_boundaries(new_bias_list1, new_weight_list1, '3', iterations1)

        set_2 = round(iterations1 / 2)

        set_2_mse = []

        for element in range(len(mse_value1)):

            if (element == set_2 - 1):
                break

            set_2_mse.append(mse_value1[element])

        set_3 = iterations1

        self.plot_learning_curve(set_2_mse[:2], 3)

        self.plot_learning_curve(set_2_mse, set_2)

        self.plot_learning_curve(mse_value1, set_3)

        #Resetting weights and bias
        self.set_weights(og_weights.copy())
        self.set_bias(og_bias)

    #Shows the three decision boundaries, learning curves, and a randomized_weights case for the given NN learner.
    def main(self):
        new_bias_list1, new_weight_list1, mse_value1, iterations1 = self.gradient_descent(0.1, 0.05)

        self.plot_decision_boundaries(new_bias_list1, new_weight_list1, '3', iterations1)
        self.plot_learning_curve(mse_value1, iterations1)

        self.randomized_weights()

weights_to_use_easy_case = [1, 3]
bias_to_use_easy_case = -9

test1 = NeuralNetworks(the_data, weights_to_use_easy_case, bias_to_use_easy_case, species)

test1.main()

weights_to_use_hard_case = [0, -1]
the_bias_to_use_hard_case = -2.23

test2 = NeuralNetworks(the_data, weights_to_use_hard_case, the_bias_to_use_hard_case, species)

#test2.main()