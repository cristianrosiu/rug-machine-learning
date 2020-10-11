import numpy as np
import pandas
import sys
import matplotlib.pyplot as plt


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def plot_epoch(X, y, prototypes):
    plt.scatter(X[:, 0], X[:, 1], c=y.tolist())
    plt.scatter(prototypes[:, 0], prototypes[:, 1], c='r')
    plt.show()


class LVQ1:
    def __init__(self, n_classes, learning_rate=0.1, prot_per_class=1):
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.prot_per_class = prot_per_class

    # Randomly initializes prototypes.
    def __random_init(self, X, y):
        prototypes = []
        prototype_labels = []

        labels = np.unique(y)

        for i in range(self.n_classes):
            for j in range(self.prot_per_class):
                prototypes.append(X[np.random.choice(X.shape[0])])
                prototype_labels.append(labels[i])

        self.prototypes = np.array(prototypes)
        self.prototype_labels = np.array(prototype_labels)

    def __shuffle_data(self, data, labels):
        randomized_data = list(zip(data.tolist(), labels.tolist()))
        np.random.shuffle(randomized_data)
        randomized_data = list(zip(*randomized_data))

        return np.array(randomized_data[0]), np.array(randomized_data[1])

    def __nearest_neighbour(self, point):
        winner = (0, sys.maxsize)
        for index, prototype in enumerate(self.prototypes):
            distance = euclidean_distance(prototype, point)
            if distance < winner[1]:
                winner = (index, distance)
        return winner[0]

    def __update_prototype(self, point, point_index, prototype_index, labels):
        sign = 1 if (self.prototype_labels[prototype_index] == labels[point_index]) else -1
        self.prototypes[prototype_index] = self.prototypes[prototype_index] + self.learning_rate * sign * (
                self.prototypes[prototype_index] - point)

    def train(self, X, y, epochs=10):
        self.__random_init(X, y)
        for epoch in range(epochs):
            points, targets = self.__shuffle_data(X, y)
            for index, point in enumerate(points):
                winner_index = self.__nearest_neighbour(point)
                self.__update_prototype(point, index, winner_index, targets)
            plot_epoch(X, y, self.prototypes)


test = pandas.read_csv('datasets/lvqdata.csv')
test = test.to_numpy()

test_labels = np.array([1 if i < 50 else 2 for i in range(test.shape[0])])

lvq = LVQ1(2, 0.1, 1)
lvq.train(test, test_labels)
