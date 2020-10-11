import numpy as np
from numpy.random import permutation
import random
import sys
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.metrics import confusion_matrix

# Create simple class used as Enum
class Status:
    def __init__(self):
        pass

    New, Visited, Noise = range(3)


def distance(a, b):
    return np.linalg.norm(a - b)


class DBscan():
    def __init__(self, points, eps=0.15, min_pts=2):
        self.points = points
        # Maximum radius of a point in order to be regarded as neighbour
        self.eps = eps
        # Minimum number of neighbours that a point needs to have in order for it's density
        # to be regarded as "high density"
        self.min_pts = min_pts
        # Number of samples / points
        self.n_pts = len(points)
        # List of labels
        self.labels = [0] * self.n_pts
        # Property Lists
        self.status_list = [Status.New] * self.n_pts
        self.member_list = [False] * self.n_pts
        self.distances = distance_matrix(self.points, self.points)

    def fit(self):
        # Initial cluster label
        cluster = 0
        for i in range(self.n_pts):
            if self.status_list[i] == Status.Visited:
                continue
            self.status_list[i] = Status.Visited
            neighbours = self.region_query(i)
            if len(neighbours) < self.min_pts:
                self.status_list[i] = Status.Noise
            else:
                cluster = cluster + 1
                self.add_to_cluster(cluster, i)
                self.expand_cluster(neighbours, cluster)

        # Set noise points in the label list as the value -1
        for i, label in enumerate(self.labels):
            if self.status_list[i] == Status.Noise:
                self.labels[i] = -1

    # Compute neighbourhood of a core point
    def region_query(self, i):
        neighbours = set()
        # Check core point's distances to all points
        for j in range(len(self.points)):
            if self.distances[i][j] <= self.eps:
                # If found point distances is in the radius
                # of the core point, add it to neighbourhood
                neighbours.add(j)
        return neighbours

    def expand_cluster(self, neighbours, cluster):
        while neighbours:
            index = neighbours.pop()
            if self.status_list[index] != Status.Visited:
                # If Noise or Undefined, transform the point into a border
                self.status_list[index] = Status.Visited
                extended_neighbours = self.region_query(index)
                if len(extended_neighbours) >= self.min_pts:
                    neighbours.update(extended_neighbours)
            if not self.member_list[index]:
                self.add_to_cluster(cluster, index)

    def add_to_cluster(self, cluster, i):
        self.labels[i] = cluster
        self.member_list[i] = True

    def evaluate(self, y_true):
        # Get the values of the confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, self.labels).ravel()
        # Compute Accuracy score.
        accuracy = float((tp + tn)/(tp + tn + fp + fn))
        # Precision and recall will help calculate the F1 score.
        precision = float(tp/(tp + fp))
        recall = float(tp/(tp + fn))
        # Print the results.
        print('Accuracy: %f' % accuracy)
        print('F1 Score: %f' % float((2 * precision * recall)/(precision+recall)))


class VQ:
    def __init__(self, K, learning_rate, epochs):
        self.K = K
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.prototypes = []
        self.squared_errors = []
        self.prototypes_trajectory = {}
        self.is_fit = False

    # Initialize the prototypes list with random points from the dataset.
    def init_prototypes(self, data):
        for i in range(self.K):
            self.prototypes.append(np.copy(data[random.randint(0, len(data))]))
            self.prototypes_trajectory[i] = []
        self.prototypes = np.array(self.prototypes)

    # Used to update a winner prototype.
    def update_prototype(self, prototype_index, point):
        self.prototypes[prototype_index] += self.learning_rate * (point - self.prototypes[prototype_index])

    # Plot all prototypes as red points.
    def plot_epoch(self, X, epoch_number, trajectory=False):
        if not self.is_fit:
            raise Exception('Use fit before you plot')
        if not trajectory:
            plt.title('Positions in Epoch:%d' % epoch_number, fontsize=20)
            plt.scatter(X[:, 0], X[:, 1], c='b')
            plt.scatter(self.prototypes[:, 0], self.prototypes[:, 1], c='r')
            plt.show()
        else:
            plt.title('Prototypes trajectory', fontsize=20)
            if epoch_number == 0:
                plt.scatter(X[:, 0], X[:, 1], c='b')
                plt.scatter(self.prototypes[:, 0], self.prototypes[:, 1], c='yellow')
            else:
                plt.scatter(self.prototypes[:, 0], self.prototypes[:, 1], c='r')

            for element in self.prototypes_trajectory.values():
                plt.plot(np.array(element)[:, 0], np.array(element)[:, 1])

    # Plots the error curve.
    def plot_error(self):
        if not self.is_fit:
            raise Exception('Use fit before you plot')
        epochs = range(self.epochs)
        plt.figure(figsize=(15, 8))
        plt.plot(epochs, self.squared_errors, 'bx-')
        plt.xlabel('EPOCH')
        plt.ylabel('SSE')
        plt.show()

    def evaluate_winner(self, data_point):
        winner = (0, sys.maxsize)
        for index, prototype in enumerate(self.prototypes):
            distance = np.linalg.norm(prototype - data_point)
            if distance < winner[1]:
                winner = (index, distance)
        return winner[0], winner[1]

    # X is the desired dataset.
    def fit(self, X, show_trajectory=False, show_plot=False):
        self.is_fit = True
        # Randomly initialize K prototypes with actual points from the dataset.
        self.init_prototypes(X)
        for epoch in range(self.epochs):
            # Change data order in every epoch.
            randomized_data = permutation(X)
            sum_squared = 0.0
            for i in range(len(self.prototypes)):
                # Keeps track of the history of prototype positions.
                self.prototypes_trajectory[i].append(list(self.prototypes[i]))
            if show_trajectory:
                self.plot_epoch(X, epoch, show_trajectory)
            if show_plot:
                self.plot_epoch(X, epoch)
            for point in randomized_data:
                # Get the closest prototype to current point.
                winner_index, distance = self.evaluate_winner(point)
                # Calculate Squared error.
                for i in range(len(point)):
                    error = point[i] - self.prototypes[winner_index][i]
                    sum_squared += error ** 2
                # Move prototype towards point using the learning_rate.
                self.update_prototype(winner_index, point)
            self.squared_errors.append(sum_squared)
        if show_trajectory:
            plt.show()
