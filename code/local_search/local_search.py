from read_data import read_data
#from .nearest_neighbour import NearestNeighbour
from .hill_climbing import two_opt
import time


import math

class NearestNeighbour:

    def __init__(self, points):
        """
        points: list of (id, x, y),
                where id is an integer 0..N-1 (or you can remap)
        """
        self.points = points
        self.dist = self.build_distance_matrix(self.points)
        self.tour, self.cost = self.nearest_neighbor_tour(self.dist)

    def build_distance_matrix(self, points):
        """
        returns: N x N matrix dist[i][j] = rounded Euclidean distance
        """
        n = len(points)
        dist = [[0]*n for _ in range(n)]

        for i in range(n):
            _, x1, y1 = points[i]
            for j in range(i+1, n):
                _, x2, y2 = points[j]
                d = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                d = int(round(d))   # round to nearest int
                dist[i][j] = d
                dist[j][i] = d      # symmetric
        return dist
    

    def nearest_neighbor_tour(self, dist, start=0):
        """
        start: starting node index (default 0)
        returns: (tour, total_cost)
                tour is a list of indices, e.g. [0, 5, 2, 3, 1]
                representing a cycle 0 -> 5 -> 2 -> 3 -> 1 -> 0
        """
        n = len(dist)
        visited = [False] * n
        tour = [start]
        visited[start] = True
        current = start

        # visit the remaining n-1 nodes
        for _ in range(n - 1):
            best_city = None
            best_dist = float('inf')

            for v in range(n):
                if not visited[v]:
                    if dist[current][v] < best_dist:
                        best_dist = dist[current][v]
                        best_city = v

            tour.append(best_city)
            visited[best_city] = True
            current = best_city

        # compute total cost
        total_cost = 0
        for i in range(n):
            total_cost += dist[tour[i]][tour[(i + 1) % n]]

        return tour, total_cost




def local_search(filename, cutoff, random_seed):
    """
    Local search using 2-opt hill climbing for the TSP.

    Parameters
    ----------
    filename : str
        Path to the .tsp file (e.g., "../data/Atlanta.tsp").
    cutoff : float
        Time cutoff in seconds.
    random_seed : int, optional
        Random seed for reproducibility (default 0).

    Returns
    -------
    tour : list
        Best tour found as vertex IDs (1-based), with start vertex repeated at end.
    """
    data = read_data(filename)  # list of (vertex_id, x, y)
    start_time = time.time()

    # build NN initial tour
    nn = NearestNeighbour(data)
    initial_tour = nn.tour          # indices
    initial_cost = nn.cost
    dist = nn.dist

    # run 2-opt hill climbing with cutoff
    best_tour, best_cost = two_opt(
        initial_tour,
        dist,
        cutoff,
        start_time
    )

    return best_tour, best_cost