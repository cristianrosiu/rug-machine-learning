import os
import numpy
import matplotlib.pyplot as plt
import math


# Summary Statistics:
#	              Min  Max   Mean    SD   Class Correlation
#   sepal length: 4.3  7.9   5.84  0.83    0.7826
#    sepal width: 2.0  4.4   3.05  0.43   -0.4194
#   petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
#    petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)

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

    results = [round(result / sample_size, 4) for result in results]
    print(results)
    print("0.6889, 0.1849, 3.0976, 0.5776")

    data_file.close()


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

    plt.figure(figsize=(8, 6))
    plt.scatter(result[:, 0], result[:, 1])

    # labeling x and y axes
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

    plt.show()

    return result


def compute_similarity_vector(data, pca_data, length):
    similarity_vector = numpy.zeros([length, 1])

    for i in range(length):
        similarity_vector[i] = (math.sqrt(sum([(a - b) ** 2 for a, b in zip(data[i], pca_data[i])])))

    print(similarity_vector)


def as_matrix():
    file_dir = os.path.dirname(os.path.abspath('__file__'))
    filename = os.path.join(file_dir, 'iris-data-set/iris.data')
    data_file = open(filename, 'r')

    matrix = []

    for line in data_file.readlines():
        line = line.split(",")
        line.pop()
        matrix.append(line)

    return numpy.array(matrix).astype(numpy.float)


compute_features_variance(sample_size=150, data_mean=numpy.array([5.84, 3.05, 3.76, 1.20]))
compute_similarity_vector(as_matrix(), pca(as_matrix(), 0.95), 150)
