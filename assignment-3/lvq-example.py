from slearning import LVQ1
import pandas
import numpy as np
import matplotlib.pyplot as plt

X = pandas.read_csv('datasets/lvqdata.csv')
X = X.to_numpy()

y = np.array([1 if i < 50 else 2 for i in range(X.shape[0])])

# Plot error
prototypes = [1, 2]
for p in prototypes:
    lvq = LVQ1(n_classes=2, learning_rate=0.002, prot_per_class=p)
    lvq.train(X=X, y=y, epochs=200, plot_error=True, stop_threshold=15)

# Plot graph
lvq = LVQ1(n_classes=2, learning_rate=0.002, prot_per_class=2)
lvq.train(X=X, y=y, epochs=200, plot_epoch=True, stop_threshold=40)
plt.scatter(X[:, 0], X[:, 1], c=lvq.predict(X))
plt.scatter(lvq.prototypes[:, 0], lvq.prototypes[:, 1], c='red')
plt.show()

