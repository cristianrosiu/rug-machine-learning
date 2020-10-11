from AssignmentAlgorithms import VQ
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas


def plot_elbow(X):
    SSE = []
    for k in range(1, 21):
        vq = VQ(k, 0.1, 100)
        vq.fit(X)
        SSE.append(sum(vq.squared_errors))
    K = range(1, 21)
    plt.figure(figsize=(15, 8))
    plt.plot(K, SSE, 'bx-')
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.show()


if __name__ == '__main__':
    X = pandas.read_csv('datasets/simplevqdata.csv').to_numpy()
    X = StandardScaler().fit_transform(X)

    # Plot the points and the error graph for various values of K and learning rate
    K = [2, 4]
    rates = [0.04, 0.1, 0.7]
    for k in K:
        for rate in rates:
            vq = VQ(k, rate, 100)
            vq.fit(X)
            vq.plot_error()

    # Plot the K vs SSE graph in order to find the most optimal K
    plot_elbow(X)
