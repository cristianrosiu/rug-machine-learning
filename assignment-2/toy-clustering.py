from ODClustering import DBscan
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


# 2.3.1 Evaluating on toy-sets
# Parameters that yield 1 cluster: We observed that if we set eps equal to min_pts, we get 1 cluster.
# 2 clusters: We observed that by increasing emps, number of clusters usually goes down. Parameters for 2 clusters
# emps = 0.36, min_pts = 2
# 3 clusters: 0.33, 2

if __name__ == "__main__":
    X = np.load('toy_set.npy')
    X = StandardScaler().fit_transform(X)

    dbscan = DBscan(X, 0.35, 6)
    dbscan.fit()
    plot_data(X, dbscan.labels)
