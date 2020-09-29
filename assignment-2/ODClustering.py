import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix


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
        # Property Lists
        self.status_list = [Status.New] * self.n_pts
        self.member_list = [False] * self.n_pts
        self.distances = distance_matrix(self.points, self.points)

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

    # Compute neighbourhood of a core point
    def region_query(self, i):
        neighbours = set()
        # Check core point's distances to all points
        for j in range(len(self.points)):
            if self.distances[i][j] <= self.eps:
                # If found point distances is in the radius
                # of the core point, add it to neighbourhood
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

