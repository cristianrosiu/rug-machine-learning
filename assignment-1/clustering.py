import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import random
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
import sys

import scipy.cluster.hierarchy as shc


def plot_pca(data_frame):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = data_frame.target.unique()
    colors = [random_color() for i in range(len(targets))]
    colors = set(colors)

    for target, color in zip(targets, colors):
        indicies_to_keep = data_frame['target'] == target
        ax.scatter(data_frame.loc[indicies_to_keep, 'principal component 1']
                   , data_frame.loc[indicies_to_keep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()


def random_color():
    r = lambda: random.randint(0, 255)
    return '#%02X%02X%02X' % (r(), r(), r())


features_data = np.load("IML_lab2_clustering/features.npy")
landmark_data = np.load("IML_lab2_clustering/gt_facialLandmarks.npy")
landmark_df = pd.DataFrame(data=landmark_data, columns=['target'])

features_data = features_data.reshape(7606, 136)

# ----------------------- DATA EXPLORATION -----------------------
x = StandardScaler().fit_transform(features_data)

pca = PCA(n_components=2)

principal_components = pca.fit_transform(x)

principal_df = pd.DataFrame(data=principal_components
                            , columns=['principal component 1', 'principal component 2'])

final_df = pd.concat([principal_df, landmark_df], axis=1)

# ----------- Part C dendogram -----------
y = pdist(x, 'cityblock')
z = linkage(y, 'single')
plt.figure(figsize=(10, 7))
sys.setrecursionlimit(10000)
p = len(final_df.target.unique())
dendrogram(z,
           truncate_mode='lastp',
           p=30,
           orientation='top',
           labels= landmark_data,
           distance_sort='descending',
           show_leaf_counts=True)
plt.show()

cluster = AgglomerativeClustering(n_clusters=4, affinity='cityblock', linkage='complete')
cluster.fit_predict(x)






