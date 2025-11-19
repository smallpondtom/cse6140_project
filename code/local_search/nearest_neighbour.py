import math

class Nearest_Neighbour:

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


