import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

"""
This is the implementation of the first exercise 1 Clustering.
Main method shows all important cases/examples to show.
"""

#Reading and shaping the data
dataStuffs = pd.read_csv("irisdata.csv")
data = dataStuffs.iloc[:, [0, 1, 2, 3]].values

data1 = dataStuffs.iloc[:, :-1]

data_for_plot = dataStuffs.iloc[:, [2,3]].values


np.random.seed(47)

#the kmeans class
class Kmeans:

    ##Constructor
    def __init__(self, k_clusters, data):
        self.k_clusters = k_clusters

        self.data = data

        self.centroids = []

        # initialize distortion values
        self.objective_func_values = []

    #Initializes the centroids for k means clustering. Used seed for replication
    def initialize_centroids(self):

        np.random.seed(120)
        self.number_of_samples, self.number_of_features = data.shape

        random_idx = np.random.choice(self.number_of_samples, self.k_clusters, replace = False)

        self.centroids = data[random_idx[:self.k_clusters]]

        return self.centroids

    #assigns k clusters to the points. Returns the cluster labels with the minimum distance
    def assign_clusters(self, some_centroids):
        distance = np.zeros((data.shape[0], self.k_clusters))
        for k in range(self.k_clusters):
            euclidean_distance = np.linalg.norm(data - some_centroids[k, :], axis=1)

            distance[:, k] = euclidean_distance
        #labels/assignments that have shortest distance for every row..identifies data with closest centroid
        cluster_label = np.argmin(distance, axis=1)
        return cluster_label


    #gets the centroids for each of the k clusters
    def get_centroids(self, labels):
        centroids = np.zeros((self.k_clusters, data.shape[1]))
        for k in range(self.k_clusters):
            #adds each centroid in k clusters into centroids list
            centroids[k, :] = np.mean(data[labels == k, :], axis = 0)
        return centroids

    #Objective function to measure distortion on each iteration
    def objective_function(self, labels, centroids):
        distance = np.zeros(data.shape[0])
        #covers the k summation
        for k in range(self.k_clusters):
            #calculates all the eulcidean distances for each k cluster
            distance[labels == k] = np.linalg.norm(data[labels == k] - centroids[k], axis=1)
        ##summation of distances...point/label summation
        distortion = np.sum(distance)
        return distortion

    """
    K mean algorithm returns list of final centroids converged and list of distortion values.
    1 a function.
    """
    def kmean_algo(self):
        #intialize centroids
        new_centroids = self.initialize_centroids().copy()

        while (True):
          #update old centroids
          old_centroids = new_centroids.copy()

          #get the new labels/assignments for k clusters
          labels = self.assign_clusters(old_centroids)

          #gets the new centroids
          new_centroids = self.get_centroids(labels)
          self.objective_func_values.append(self.objective_function(labels, new_centroids))

          #break if there is convergence
          if np.all(old_centroids == new_centroids):
             break

        for i in self.objective_func_values:
            print(i)
        print("old cent")
        print(old_centroids)
        print("new cent")
        print(new_centroids)

        copy_centroids =old_centroids.copy()
        copy_objective_func_values = self.objective_func_values.copy()

        self.centroids = []
        self.objective_func_values = []

        return copy_centroids, copy_objective_func_values

    #1 b function. Plots the objective function given an array of distortion values
    def plot_objective_function(self, obj_funct_values):
        plt.xlabel('iteration')
        plt.ylabel('objective function')
        plt.plot(obj_funct_values, "o")
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()

    #1 C function. Plots the centroid locations on each iteration. Orange X's represent centroids
    def plot_centroids(self):
        fig, ax = plt.subplots()
        iteration_number = 0

        plt.xlabel('petal length')
        plt.ylabel('petal width')
        plt.title('Iteration 0')

        new_centroids = self.initialize_centroids().copy()

        while (True):
            ##Plot entroids here
            ##plot_centroids()
            plt.xlabel('petal length')
            plt.ylabel('petal width')
            plt.title('Iteration number {:d}'.format(iteration_number))
            plt.plot(data[:, 2], data[:, 3], 'o')
            plt.plot(new_centroids[:, 2], new_centroids[:, 3], 'x')

            old_centroids = new_centroids.copy()
            # Old centroids are k cluster centers
            labels = self.assign_clusters(old_centroids)
            new_centroids = self.get_centroids(labels)

            self.objective_func_values.append(self.objective_function(labels, new_centroids))
            plt.show()
            plt.clf()
            plt.cla()
            plt.close()
            print(iteration_number)
            if np.all(old_centroids == new_centroids):
                self.objective_func_values.append(self.objective_function(labels, new_centroids))
                self.centroids = []
                self.objective_func_values = []
                plt.clf()
                plt.cla()
                plt.close()
                break

            iteration_number += 1

    """
    1 Part D plots decision boundary for either k = 2 or 3 depending on parameter.
    Relies on the use of the midpoint between centroids and finding the orthogonal line to separate them.
    This is an adequate and one of the only ways of implementing decision boundaries for k clusters
    """
    def plot_kmeans_decisionboundary(self, list_of_centroids, k_clusters):
        #figure, ax = plt.subplots()
        plt.xlabel('petal length')
        plt.ylabel('petal width')
        plt.plot(data[:, 2], data[:, 3], 'o')
        plt.plot(list_of_centroids[:, 2], list_of_centroids[:, 3], 'x')

        spatial_controllers = list_of_centroids[:, 2:4]

        #Case for k = 2
        if (k_clusters == 2):

            centroids_to_use = list_of_centroids

            p2 = centroids_to_use[0]
            p1 = centroids_to_use[1]

            slope = ((p2[3] - p1[3])) / ((p2[2] - p1[2]))

            perp_slope = -1 / (((p2[2] - p1[2])) / ((p2[3] - p1[3])))

            midpoint_y = (p2[3] + p1[3]) / 2

            midpoint_x = (p2[2] + p1[2]) / 2


            # intercept = midpoint_x * slope - midpoint_y
            intercept = -midpoint_x * perp_slope + midpoint_y

            x_values = range(0,15)

            y_values = perp_slope * x_values + intercept


            plt.plot(x_values, y_values, color = "green")

            plt.ylim(0, 3)
            plt.xlim(0, 7)

            plt.show()
            plt.clf()
            plt.cla()
            plt.close('all')

            #plt.figure

        #Case for when k = 3
        elif(k_clusters == 3):

            centroids_to_use = list_of_centroids

            p3 = centroids_to_use[2]
            p2 = centroids_to_use[0]
            p1 = centroids_to_use[1]

            perp_slope_for_1_2 = -1 / (((p2[2] - p1[2])) / ((p2[3] - p1[3])))

            perp_slope_for_1_3 = -1 / (((p3[2] - p1[2])) / ((p3[3] - p1[3])))

            perp_slope_for_2_3 = -1 / (((p3[2] - p2[2])) / ((p3[3] - p2[3])))

            midpoint_y_1_2 = (p2[3] + p1[3]) / 2

            midpoint_x_1_2 = (p2[2] + p1[2]) / 2

            midpoint_y_1_3 = (p3[3] + p1[3]) / 2

            midpoint_x_1_3 = (p3[2] + p1[2]) / 2

            midpoint_y_3_2 = (p2[3] + p3[3]) / 2

            midpoint_x_3_2 = (p2[2] + p3[2]) / 2

            intercept_1_2 = -midpoint_x_1_2 * perp_slope_for_1_2 + midpoint_y_1_2

            intercept_1_3 = -midpoint_x_1_3  * perp_slope_for_1_3 + midpoint_y_1_3

            intercept_2_3 = -midpoint_x_3_2 * perp_slope_for_2_3 + midpoint_y_3_2

            x_values = range(0, 15)

            y_values_1_2 = perp_slope_for_1_2 * x_values + intercept_1_2

            y_values_1_3 = perp_slope_for_1_3 * x_values + intercept_1_3

            #y_values_3_2 = perp_slope_for_2_3 * x_values + intercept_2_3

            plt.plot(x_values, y_values_1_2, color="green", label = 'boundary centroid one and two')

            plt.plot(x_values, y_values_1_3, color="red", label = 'boundary centroid one and three')

            plt.legend()
            plt.xlim(0,7)
            plt.ylim(0, 3)

            plt.show()
            plt.clf()
            plt.cla()
            plt.close('all')

        plt.show()
        plt.clf()
        plt.cla()
        plt.close('all')


    #Main method to show the critical implementation and writeup
    def main(self):

        centroids_to_use, obj_values = self.kmean_algo()

        self.plot_objective_function(obj_values)

        self.plot_centroids()
        plt.clf()
        plt.cla()
        plt.close()

        self.plot_kmeans_decisionboundary(centroids_to_use, self.k_clusters)

test = Kmeans(2, data)

test.main()

test2 = Kmeans(3, data)

test2.main()

