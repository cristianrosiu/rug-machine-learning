from AssignmentAlgorithms import DBscan
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def plot_data(data, labels):
    fig, ax = plt.subplots()

    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='nipy_spectral', s=15)
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)
    plt.show()


if __name__ == "__main__":
    X = np.load('datasets/toy_set.npy')
    X = StandardScaler().fit_transform(X)

    eps_values = [0.15, 0.2, 0.25, 0.3, 0.32, 0.35, 0.4, 0.45, 0.5]
    min_pts_values = [3, 4, 5, 6, 7, 8, 9, 10]
    for epsilon in eps_values:
        for min_samples in min_pts_values:
            dbscan = DBscan(X, epsilon, min_samples)
            dbscan.fit()
            plot_data(X, dbscan.labels)
            print(silhouette_score(X, dbscan.labels))
