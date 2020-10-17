from slearning import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


def plot_error(X, y, X_test, y_test):
    P = range(30, 500, 10)
    train_error = []
    test_error = []
    for size in P:
        lr = LinearRegression()
        lr.train(X[:size], y[:size])
        train_error.append(lr.MSE(X, y))
        test_error.append(lr.MSE(X_test, y_test))

    plt.figure(figsize=(15, 8))
    plt.plot(P, train_error, 'bx-')
    plt.plot(P, test_error, 'bx-', color='r')
    plt.legend(('MSE Train', 'MSE Test'))
    plt.xlabel('P')
    plt.ylabel('MSE')
    plt.show()


def plot_weights(X, y, X_test, y_test):
    P = [30, 40, 50, 75, 100, 200, 300, 400, 500]

    for size in P:
        lr = LinearRegression()
        lr.train(X[:size], y[:size])

        x_values = range(25)
        plt.bar(x_values, list(lr.weights))
        plt.show()


if __name__ == '__main__':
    X = np.genfromtxt('datasets/xtrain.csv', delimiter=',', skip_header=True)
    y = np.genfromtxt('datasets/ytrain.csv', delimiter=',', skip_header=True)

    X_test = np.genfromtxt('datasets/xtest.csv', delimiter=',', skip_header=True)
    y_test = np.genfromtxt('datasets/ytest.csv', delimiter=',', skip_header=True)

    plot_error(X, y, X_test, y_test)
    plot_weights(X, y, X_test, y_test)
