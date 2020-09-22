import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as shc
import sys


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


def freq_bar(data, data_frequency, title, x_label, y_label):
    plt.bar(data, data_frequency)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# Computes the list of sums of Squared Errors for all the points (WSS)
def elbow_method(data, k_max):
    # Compute WSS of each
    sse = []
    for k in range(1, k_max + 1):
        k_means = KMeans(n_clusters=k)
        k_means = k_means.fit(data)
        sse.append(k_means.inertia_)

    # Plot WSS vs K
    K = range(1, k_max + 1)
    plt.plot(K, sse, 'bx-')
    plt.xlabel('K')
    plt.ylabel('Sum Of Squared Distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


# Computes WSS/SSE for a specific number of clusters K
def compute_internal_measure(data, clusters_labels, centroids, measure):
    data_center = np.mean(data, axis=0)
    sum = 0

    for i in range(len(data)):
        # Get the list of means for the a specific data point (center of a cluster).
        center = centroids[clusters_labels[i]]
        for j in range(len(data[1])):
            # Compute either wss or bss for each data point and add it to the sum
            if measure == 'wss':
                sum = sum + (data[i, j] - center[j]) ** 2
            elif measure == 'bss':
                sum = sum + clusters_labels.count(clusters_labels[i]) * (data_center[j] - center[j]) ** 2
            elif measure == 'sse':
                sum = sum + (data[i,j] - data_center[j]) **2
            else:
                raise Exception("Wrong input for measure")
    # Final sum is the wss/bss value of a specific cluster.
    return sum

def perf_measure(true_labels, predicted_labels):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(true_labels) - 1):
        for j in range(i, len(true_labels)):
            if true_labels[i] == true_labels[j] and predicted_labels[i] == predicted_labels[j]:
                TP += 1
            elif true_labels[i] == true_labels[j] and predicted_labels[i] != predicted_labels[j]:
                FN += 1
            elif true_labels[i] != true_labels[j] and predicted_labels[i] == predicted_labels[j]:
                FP += 1
            elif true_labels[i] != true_labels[j] and predicted_labels[i] != predicted_labels[j]:
                TN += 0

    return TP, FP, TN, FN

# Apply dimensionality reduction on a dataset
def d_reduction(n_comp, targets):
    pca = PCA(n_components=n_comp)
    # Apply PCA Algorithm in order to reduce the dimension to the desired number.
    principal_components = pca.fit_transform(x)
    # Convert result to pandas data frame
    principal_df = pd.DataFrame(data=principal_components
                                , columns=['principal component 1', 'principal component 2'])

    # Add targets column for better visualization of data
    final_df = pd.concat([principal_df, targets], axis=1)

    # Returns new data as a panda dataframe
    return final_df

def h_clustering_analysis(x, k):
    linkages = ['single', 'complete', 'average', 'ward']

    for link in linkages:
        print("Linkage:", link.upper(), "Affinity:", 'cityblock' if link.lower() != 'WARD' else 'EUCLIDEAN',
              "Number of clusters:", k)
        model = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage=link) if link.lower() == 'ward' \
                                        else AgglomerativeClustering(n_clusters=k, affinity='cityblock', linkage=link)
        pred_labels = model.fit_predict(x)
        cluster_labels = model.labels_

        clf = NearestCentroid()
        clf.fit(x, pred_labels)

        wss = compute_internal_measure(x, cluster_labels, clf.centroids_, 'wss')
        sse = compute_internal_measure(x, cluster_labels, clf.centroids_, 'sse')
        bss = sse - wss
        print("WSS:", wss, "BSS:", bss)

        # ----------- Part E -----------
        TP, FP, TN, FN = perf_measure(landmark_data, cluster_labels)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f_score = (2 * precision * recall) / (precision + recall)
        print("Accuracy", accuracy)
        print("Precision", precision)
        print("Recall", recall)
        print("F score", f_score)
        print()
        print("-----------------------------------------------")
        print()


features_data = np.load("IML_lab2_clustering/features.npy")
landmark_data = np.load("IML_lab2_clustering/gt_facialLandmarks.npy")
landmark_df = pd.DataFrame(data=landmark_data, columns=['target'])

features_data = features_data.reshape(7606, 136)

# ----------------------- DATA EXPLORATION -----------------------
x = StandardScaler().fit_transform(features_data)

# Bar plot for better visualisation of the data
unique_labels = landmark_df['target'].unique().tolist()
labels_freq = [landmark_data.tolist().count(label) for label in unique_labels]
freq_bar(unique_labels, labels_freq, "Landmarks Frequency Bar Plot", "Landmark", "Frequency")

# Reduce data to 2 dimensions
reduced_data = d_reduction(2, landmark_df)
reduced_data = reduced_data[reduced_data['target'] < 21]
# Plot newly reduced data
plot_pca(reduced_data)


# ----------- Part C + D -----------
# Try H Clustering with K = 5
h_clustering_analysis(x, 5)

# Apply elbow method to check which K is Optimal
elbow_method(x, 5)

# Try H Clustering with optimal K
h_clustering_analysis(x, 3)










