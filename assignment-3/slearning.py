import numpy as np
import sys
import matplotlib.pyplot as plt


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


class LVQ1:
    def __init__(self, n_classes, learning_rate=0.1, prot_per_class=1):
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.prot_per_class = prot_per_class
        self.__trained = False

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

    def __calculate_error(self, X, y):
        error_sum = 0
        y_pred = self.predict(X)
        for i, label in enumerate(y_pred):
            if label != y[i]:
                error_sum += 1

        return error_sum / 100

    def __plot_epoch(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y.tolist())
        plt.scatter(self.prototypes[:, 0], self.prototypes[:, 1], c='r')
        plt.show()

    def __plot_error(self, epochs, errors):
        K = range(1, epochs + 1)
        plt.figure(figsize=(15, 8))
        plt.plot(K, errors, 'bx-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Error in %')
        plt.show()

    def train(self, X, y, epochs=10, plot_error=False, plot_epoch=False, stop_threshold=10):
        self.__random_init(X, y)
        errors = []
        prev_error = 0
        duplicates = 0
        for epoch in range(epochs):
            if duplicates >= stop_threshold:
                epochs = epoch
                break
            points, targets = self.__shuffle_data(X, y)
            for index, point in enumerate(points):
                winner_index = self.__nearest_neighbour(point)
                self.__update_prototype(point, index, winner_index, targets)
            errors.append(self.__calculate_error(X, y))
            if epoch == 0:
                prev_error = errors[0]
            else:
                current = errors[len(errors) - 1]
                if prev_error == current:
                    duplicates += 1
                else:
                    duplicates = 0
                prev_error = current
        if plot_epoch:
            self.__plot_epoch(X, y)
        self.__trained = True
        if plot_error:
            self.__plot_error(epochs, errors)

    def predict(self, samples):
        predicted_labels = []
        for point in samples:
            predicted_labels.append(self.prototype_labels[self.__nearest_neighbour(point)])
        return predicted_labels


class LinearRegression:
    def __init__(self):
        self.weights = np.array([])

    def MSE(self, X, y):
        sum = 0

        for i, point in enumerate(X):
            sum = sum + pow(np.dot(self.weights, point) - y[i], 2)

        return sum / (2 * len(X))

    def train(self, X, y):
        p_inverse = np.linalg.pinv(X.T.dot(X))
        self.weights = p_inverse.dot(X.T.dot(y))
