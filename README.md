# NeuralNetworksFromScratch
The following repository contains various types of neural networks implemented from scratch.

**1. learner.py:** implements gradient descent to learn a desicision boundary on iris data set on classes of versicolor and virginica. This is done via a single-layer classifier "perceptron". Includes plots of optimizing decision boundary and iterations vs learning curves. Shows use of random wieghts vs preset weights.

**2. FNN.py:** implements a basic feedforward neural network architecture to solve the XOR problem. It allows for one to create FNN of any size. Uses gradient descent with a sigmoid activation function...thus loss function is derviative of sigmoid. Error measure is mean squared error. Provides a average loss and success rate statistics. 

**3. kmean.py:** implements the supervised machine learning clustering algorithm, kmeans, on the iris data set. Includes additional elements as it was part of course project in univeristy. Includes plotting the objective function as a function for each iteration. Plots the process of the learning algorithm (initial, intermediate, converged) for k = 2 and k = 3. Also, plots the decision boundary(ies) for optimized paramters. 


