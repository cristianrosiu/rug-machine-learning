import os
import numpy
import matplotlib.pyplot as plt



# Summary Statistics:
#	              Min  Max   Mean    SD   Class Correlation
#   sepal length: 4.3  7.9   5.84  0.83    0.7826
#    sepal width: 2.0  4.4   3.05  0.43   -0.4194
#   petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
#    petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)


def pca(data, alpha):
    mean = data.mean(axis=0)
    center_data = data - mean
    covariance = numpy.dot(numpy.transpose(center_data), center_data) / 150

    eigenvalues, eigenvectors = numpy.linalg.eig(covariance)
    eigenvalue_sum = numpy.sum(eigenvalues)

    eigenvalues = eigenvalues.tolist()
    print("eigenvalues")
    print(eigenvalues)

    f_r = 0
    r = 0
    while f_r < alpha:
        f_r = f_r + (eigenvalues.pop(0)/eigenvalue_sum)
        r = r + 1

    eigenvectors = numpy.transpose(eigenvectors)
    eigenvectors = eigenvectors[0:r]

    print("eigenvectors")
    print(eigenvectors)

    A = []

    #for i in range(150):
     #   for j in range(4):
    #        A.append(numpy.multiply(data[i][j], eigenvectors))

    A = numpy.dot(data, numpy.transpose(eigenvectors))
    print(A)

    fig, ax = plt.subplots()
    colors = list('bgrcmykw')

    for i in range(150):
        x = A[i][0]
        y= A[i][1]
        ax.scatter(x, y)

    plt.show()


def as_matrix():
    file_dir = os.path.dirname(os.path.abspath('__file__'))
    filename = os.path.join(file_dir, 'assignment-1/iris-data-set/iris.data')
    data_file = open(filename, 'r')

    matrix = []

    for line in data_file.readlines():
        line = line.split(",")
        line.pop()
        matrix.append(line)

    return numpy.array(matrix).astype(numpy.float)


# sample_size = 150
# data_mean = numpy.array([5.84, 3.05, 3.76, 1.20])

pca(as_matrix(), 0.95)

# fileDir = os.path.dirname(os.path.relpath('__file__'))
# filename = os.path.join(fileDir, 'iris-data-set/iris.data')

# data_file = open(filename, 'r')
# results = [0 for i in range(4)]

# for line in data_file.readlines():
#    line = line.split(",")
#    line.pop()

#     results[0] += (float(line[0]) - sepal_length_mean) ** 2
#     results[1] += (float(line[1]) - sepal_width_mean) ** 2
#     results[2] += (float(line[2]) - petal_length_mean) ** 2
#     results[3] += (float(line[3]) - petal_width_mean) ** 2
#
# results = [round(result/sample_size,4) for result in results]
#
# print(results)
# print("0.6889, 0.1849, 3.0976, 0.5776")
#
# data_file.close()
