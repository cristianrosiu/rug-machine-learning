from AssignmentAlgorithms import VQ
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas

if __name__ == '__main__':
    X = pandas.read_csv('simplevqdata.csv').to_numpy()
    X = StandardScaler().fit_transform(X)

    vq = VQ(2, 0.05, 300)
    vq.fit(X)
    epochs = range(300)
    plt.figure(figsize=(15, 8))
    plt.plot(epochs, vq.squared_errors, 'bx-')
    plt.xlabel('EPOCH')
    plt.ylabel('SSE')
    plt.show()
