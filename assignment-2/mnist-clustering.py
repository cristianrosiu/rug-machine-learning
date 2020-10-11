from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import matplotlib.pyplot as plt
from AssignmentAlgorithms import DBscan
from sklearn.neighbors import NearestNeighbors
import numpy as np


def plot_data(data, labels):
    fig, ax = plt.subplots()

    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='nipy_spectral', s=20)
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)
    plt.show()


mnist_data = sio.loadmat('datasets/mnist.mat')

# --------- Data Exploration ---------
X = mnist_data['X']
y = mnist_data['y']
# Standardize the data
X = StandardScaler().fit_transform(X)
# Apply PCA on given data
pca = PCA(n_components=2)
X = pca.fit_transform(X)
# Plot the data and color code using the given labels (ground truth)
plt.scatter(X[:, 0], X[:, 1], c=y, s=5)
plt.show()

# --------- Clustering - Outlier Detection ---------

# Find optimal value for Epsilon and plot the graph
neigh = NearestNeighbors(n_neighbors=4)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.xlabel('Points Sorted According to Distance of 4th Nearest Neighbour')
plt.ylabel('4th Nearest Neighbour Distance')
plt.grid()
plt.plot(distances)

# Apply algorithm using the optimal value
dbs = DBscan(X, eps=0.35, min_pts=20)
dbs.fit()
y_pred = dbs.labels
plot_data(X, y_pred)
# Transform -1 labels to 0 in order to mach the ground-truth data
for i in range(len(y_pred)):
    if y_pred[i] == -1:
        y_pred[i] = 0
# Evaluate Scores
dbs.evaluate(list(y))





