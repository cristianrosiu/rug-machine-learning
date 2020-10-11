import numpy as np
import random


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


class LVQ1:
    def __init__(self, n_inputs, n_classes, learning_rate=0.1, prot_per_class=1):
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.prot_per_class = prot_per_class
        self.prototypes = np.empty((0, n_inputs))

    def __init_prototypes(self, X):
        random_index = random.randint(0, len(X) - 1)
        for i in range(self.n_classes):
            self.prototypes = np.append(self.prototypes, np.array([X[random_index]]), axis=0)

    def train(self, X, y, epoch=10):
        self.__init_prototypes(X)

        pass
