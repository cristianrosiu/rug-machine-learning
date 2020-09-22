import os
import numpy
import matplotlib.pyplot as plt
import math
from itertools import combinations
from itertools import product
from sklearn.decomposition import PCA


# Compute variance per plant feature.
def compute_features_variance(sample_size, data_mean):
    fileDir = os.path.dirname(os.path.relpath('__file__'))
    filename = os.path.join(fileDir, 'iris-data-set/iris.data')

    data_file = open(filename, 'r')
    results = [0 for i in range(4)]

    for line in data_file.readlines():
        line = line.split(",")
        line.pop()

        results[0] += (float(line[0]) - data_mean[0]) ** 2
        results[1] += (float(line[1]) - data_mean[1]) ** 2
        results[2] += (float(line[2]) - data_mean[2]) ** 2
        results[3] += (float(line[3]) - data_mean[3]) ** 2

    # Calculate Variance
    results = [round(result / sample_size, 4) for result in results]
    data_file.close()

    return results


def compute_variance_ratio(eigenvalues, alpha):
    eigenvalue_sum = numpy.sum(eigenvalues)

    ratio = 0
    # Dimensionality number.
    r = 0
    while ratio < alpha:
        ratio = ratio + (eigenvalues.pop(0) / eigenvalue_sum)
        r = r + 1
    return ratio, r


def pca(data, alpha):
    # Get mean of values and center data accordingly
    mean = data.mean(axis=0)
    center_data = data - mean

    # Calculate the covariance
    covariance = numpy.dot(numpy.transpose(center_data), center_data) / 150

    # Get the Eigenvalues & Eigenvectors
    eigenvalues, eigenvectors = numpy.linalg.eig(covariance)

    eigenvalues = eigenvalues.tolist()

    # Compute the ratio of total variance
    variance_ratio, r = compute_variance_ratio(eigenvalues, alpha)

    # Do dimensionality reduction
    eigenvectors = numpy.transpose(eigenvectors)
    eigenvectors = eigenvectors[0:r]

    result = numpy.dot(center_data, numpy.transpose(eigenvectors))

    return result


# Transform data frame to numpy array
def data_to_numpy():
    file_dir = os.path.dirname(os.path.abspath('__file__'))
    filename = os.path.join(file_dir, 'iris-data-set/iris.data')
    data_file = open(filename, 'r')

    matrix = []

    for line in data_file.readlines():
        line = line.split(",")
        line.pop()  # Remove names from array.
        matrix.append(line)

    return numpy.array(matrix).astype(numpy.float)


def average_euclidean_distance(points):
    esum = 0
    for tuple in points:
        esum = esum + math.sqrt(sum([(a - b) ** 2 for a, b in zip(tuple[0], tuple[1])]))
    return esum / len(points)


def plot_pca(array, dimension):
    plt.figure()

    if dimension == 2:  # 2D Plot
        plt.scatter(array[:50, 0], array[:50, 1], color='red')
        plt.scatter(array[50:100, 0], array[50:100, 1], color='green')
        plt.scatter(array[100:150, 0], array[100:150, 1], color='blue')

        # labeling x and y axes
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')

    elif dimension == 3:  # 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.scatter(array[:50, 0], array[:50, 1], array[:50, 2], color='red', marker='o')
        plt.scatter(array[50:100, 0], array[50:100, 1], array[50:100, 2], color='green', marker='o')
        plt.scatter(array[100:150, 0], array[100:150, 1], array[100:150, 2], color='blue', marker='o')

        # labeling x, y and z axes
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')

    plt.show()


# ---------------------------------------- Data Exploration ----------------------------------------
print("------ Data Exploration ------")
print("Variance Per Feature: ")
variance_array = compute_features_variance(sample_size=150, data_mean=numpy.array([5.84, 3.05, 3.76, 1.20]))
print("Sepal Length Variance: ", variance_array[0])
print("Sepal Width Variance: ", variance_array[1])
print("Petal Length Variance: ", variance_array[2])
print("Petal Width Variance: ", variance_array[3], '\n')

# ---------------------------------------- Data Analysis: PCA ----------------------------------------

data = data_to_numpy()

# Compute PCA for 3 different dimensions using our algorithm.
PC_1 = pca(data, 0.92)
PC_2 = pca(data, 0.93)
PC_3 = pca(data, 0.98)
PC_4 = pca(data, 1)

# Compute PCA result using sklearn library.
pca_sklearn = PCA(n_components=2)
transformed_data = pca_sklearn.fit_transform(data)

# Plot both PCA results so we can compare them. PART D
plot_pca(PC_2, 2)
plot_pca(transformed_data, 2)

# ---------------------------------------- Dimensionality reduction evaluation ----------------------------------------
print("------ Dimensionality reduction evaluation ------")

# Compute euclidean distance per class and print it.
# Euclidean distance for 1 PC
print("Average Euclidean Distances 1 PC:")
print(average_euclidean_distance(list(combinations(PC_1[:50], 2))))
print(average_euclidean_distance(list(combinations(PC_1[50:100], 2))))
print(average_euclidean_distance(list(combinations(PC_1[100:150], 2))), '\n')

# Euclidean distance for 2 PC
print("Average Euclidean Distances 2 PC:")
print(average_euclidean_distance(list(combinations(PC_2[:50], 2))))
print(average_euclidean_distance(list(combinations(PC_2[50:100], 2))))
print(average_euclidean_distance(list(combinations(PC_2[100:150], 2))), '\n')

# Euclidean distance for 3 PC
print("Average Euclidean Distances 3 PC:")
print(average_euclidean_distance(list(combinations(PC_3[:50], 2))))
print(average_euclidean_distance(list(combinations(PC_3[50:100], 2))))
print(average_euclidean_distance(list(combinations(PC_3[100:150], 2))), '\n')

# Euclidean distance for 4 PC
print("Average Euclidean Distances 4 PC:")
print(average_euclidean_distance(list(combinations(PC_4[:50], 2))))
print(average_euclidean_distance(list(combinations(PC_4[50:100], 2))))
print(average_euclidean_distance(list(combinations(PC_4[100:150], 2))), '\n')

# Euclidean distance between each individual 1 PC component
print("Average Euclidean Distances 1 PC between each different component")
print(average_euclidean_distance(list(product(PC_1[:50], PC_1[50:100]))))
print(average_euclidean_distance(list(product(PC_1[:50], PC_1[100:150]))))
print(average_euclidean_distance(list(product(PC_1[50:100], PC_1[100:150]))), '\n')

# Euclidean distance between each individual 2 PC component
print("Average Euclidean Distances 2 PC between each different component")
print(average_euclidean_distance(list(product(PC_2[:50], PC_2[50:100]))))
print(average_euclidean_distance(list(product(PC_2[:50], PC_2[100:150]))))
print(average_euclidean_distance(list(product(PC_2[50:100], PC_2[100:150]))), '\n')

# Euclidean distance between each individual 3 PC component
print("Average Euclidean Distances 3 PC between each different component")
print(average_euclidean_distance(list(product(PC_3[:50], PC_3[50:100]))))
print(average_euclidean_distance(list(product(PC_3[:50], PC_3[100:150]))))
print(average_euclidean_distance(list(product(PC_3[50:100], PC_3[100:150]))), '\n')

# Euclidean distance between each individual 4 PC component
print("Average Euclidean Distances 4 PC between each different component")
print(average_euclidean_distance(list(product(PC_4[:50], PC_4[50:100]))))
print(average_euclidean_distance(list(product(PC_4[:50], PC_4[100:150]))))
print(average_euclidean_distance(list(product(PC_4[50:100], PC_4[100:150]))), '\n')

# Compare 3D graphs of both algorithms
pca_sklearn = PCA(n_components=3)
transformed_data = pca_sklearn.fit_transform(data)

plot_pca(PC_3, 3)
plot_pca(transformed_data, 3)
