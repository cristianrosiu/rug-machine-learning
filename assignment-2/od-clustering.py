import numpy as np
from sklearn.cluster import DBSCAN


# Create simple class used as Enum
class Status:
    def __init__(self):
        pass

    New, Visited, Noise = range(3)


def distance(a, b):
    return np.linalg.norm(a - b)


class DBscan():
    def __init__(self, points, eps=0.15, min_pts=2):
        self.points = points
        # Maximum radius of a point in order to be regarded as neighbour
        self.eps = eps
        # Minimum number of neighbours that a point needs to have in order for it's density
        # to be regarded as "high density"
        self.min_pts = min_pts
        # Number of samples / points
        self.n_pts = len(points)
        # List of labels
        self.labels = [0] * self.n_pts
        # List of status.
        self.status_list = [Status.New] * self.n_pts
        self.member_list = [False] * self.n_pts

    def fit(self):
        # Initial cluster label
        cluster = 0
        for i in range(self.n_pts):
            if self.status_list[i] == Status.Visited:
                continue
            self.status_list[i] = Status.Visited
            neighbours = self.region_query(i)
            if len(neighbours) < self.min_pts:
                self.status_list[i] = Status.Noise
            else:
                cluster = cluster + 1
                self.add_to_cluster(cluster, i)
                self.expand_cluster(neighbours, cluster)

        # Set noise points in the label list as the value -1
        for i, label in enumerate(self.labels):
            if self.status_list[i] == Status.Noise:
                self.labels[i] = -1

    def region_query(self, i):
        neighbours = set()
        for j, other_point in enumerate(self.points):
            if distance(self.points[i], other_point) <= self.eps:
                neighbours.add(j)
        return neighbours

    def expand_cluster(self, neighbours, cluster):
        while neighbours:
            index = neighbours.pop()
            if self.status_list[index] != Status.Visited:
                # If Noise or Undefined, transform the point into a border
                self.status_list[index] = Status.Visited
                extended_neighbours = self.region_query(index)
                if len(extended_neighbours) >= self.min_pts:
                    neighbours.update(extended_neighbours)
            if not self.member_list[index]:
                self.add_to_cluster(cluster, index)

    def add_to_cluster(self, cluster, i):
        self.labels[i] = cluster
        self.member_list[i] = True


if __name__ == "__main__":
    data = np.load('toy_set.npy')

    dbscan = DBscan(data)
    dbscan.fit()

    print(dbscan.labels)
