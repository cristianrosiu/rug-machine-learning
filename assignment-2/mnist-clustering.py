from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import matplotlib.pyplot as plt
from AssignmentAlgorithms import DBscan


def plot_data(data, labels):
    fig, ax = plt.subplots()

    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='nipy_spectral', s=15)
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)
    plt.show()


mnist_data = sio.loadmat('mnist.mat')

# --------- Data Exploration ---------
X = mnist_data['X']
y = mnist_data['y']
# Standardize the data
X = StandardScaler().fit_transform(X)
# Apply PCA on given data
pca = PCA(n_components=2)
X = pca.fit_transform(X)
# Plot the data and color code using the given labels (ground truth)
# REPORT: From what we can observe, there are 2 different clusters that are related to
# one another
# plt.scatter(X[:, 0], X[:, 1], c=y, s=5)
# plt.show()

# --------- Clustering - Outlier Detection ---------
eps_values = [0.1, 0.15, 0.2, 0.25]
for i, value in enumerate(eps_values):
    dbs = DBscan(X, eps=value, min_pts=5)
    dbs.fit()
    y_pred = dbs.labels
    plot_data(X, y_pred)
