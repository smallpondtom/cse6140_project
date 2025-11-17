#!/usr/bin/env python3

"""
This function runs brute-force (BF) algorithm for Traveling Salesman Problem

inputs
  - fname: filename of a dataset (path to .tsp file)
  - coff: time cut-off (sec) to terminate the algorithm

outputs
  - current_tour: best tour found (sequence of vertex IDs, with start repeated at end)
  - opt_d: total distance of the best tour (rounded to nearest integer)

This implementation:
  - uses read_data.py to read the TSP coordinates
  - builds an integer-rounded Euclidean distance matrix
  - runs an exact branch-and-bound search with a time cutoff
  - always returns the best tour found so far when the cutoff is reached
"""

import time
import math
import numpy as np
from read_data import read_data

class BruteForceTSPSolver:
    """
    Exact TSP solver using depth-first branch-and-bound with a time cutoff.

    data: list of (vertex_id, x, y) from read_data(fname)
    cutoff_seconds: time limit in seconds
    """

    def __init__(self, data, cutoff_seconds):
        self.data = data
        self.n = len(data)
        self.cutoff = cutoff_seconds

        self.start_time = None
        self.best_cost = math.inf
        self.best_tour_indices = None  # tour as indices in [0, n-1]
        self.timed_out = False

        # Precompute distance matrix (integer-rounded Euclidean distances)
        self.dist = self._build_distance_matrix()

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def _build_distance_matrix(self):
        """Build an n x n symmetric distance matrix from (x, y) coordinates."""
        n = self.n
        dist = [[0] * n for _ in range(n)]
        coords = [(x, y) for (_, x, y) in self.data]

        for i in range(n):
            xi, yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                w = int(round(d))
                dist[i][j] = dist[j][i] = w
        return dist

    def _time_exceeded(self):
        return (time.time() - self.start_time) >= self.cutoff

    def _nearest_neighbor_tour(self, start=0):
        """
        Simple nearest-neighbor heuristic used only to get an initial
        upper bound for branch-and-bound.
        """
        n = self.n
        visited = {start}
        path = [start]
        cost = 0
        current = start

        while len(path) < n:
            best_j = None
            best_d = math.inf
            for j in range(n):
                if j in visited:
                    continue
                d = self.dist[current][j]
                if d < best_d:
                    best_d = d
                    best_j = j

            visited.add(best_j)
            path.append(best_j)
            cost += best_d
            current = best_j

        # close the tour
        cost += self.dist[current][start]
        return path, cost

    def _dfs(self, current, visited, path, cost_so_far):
        """Depth-first search with branch-and-bound and time cutoff."""
        # Check time cutoff
        if self._time_exceeded():
            self.timed_out = True
            return

        # If we've visited all vertices, close the tour and update best if needed
        if len(path) == self.n:
            total_cost = cost_so_far + self.dist[current][path[0]]
            if total_cost < self.best_cost:
                self.best_cost = total_cost
                self.best_tour_indices = list(path)
            return

        # Explore remaining vertices ordered by increasing outgoing edge cost
        candidates = [j for j in range(self.n) if j not in visited]
        candidates.sort(key=lambda j: self.dist[current][j])

        for nxt in candidates:
            new_cost = cost_so_far + self.dist[current][nxt]

            # Branch-and-bound: prune if partial cost already worse than best
            if new_cost >= self.best_cost:
                continue

            visited.add(nxt)
            path.append(nxt)
            self._dfs(nxt, visited, path, new_cost)
            path.pop()
            visited.remove(nxt)

            if self.timed_out:
                return

    def solve(self, start=0):
        """
        Run the search and return (best_tour_indices, best_cost, timed_out).

        best_tour_indices is a list of indices in [0, n-1].
        best_cost is the tour length (not yet rounded).
        """
        self.start_time = time.time()

        # 1) Get an initial upper bound from nearest neighbor
        nn_tour, nn_cost = self._nearest_neighbor_tour(start)
        self.best_tour_indices = nn_tour
        self.best_cost = nn_cost

        # 2) Exact search with branch-and-bound
        visited = {start}
        path = [start]
        self._dfs(start, visited, path, 0)

        return self.best_tour_indices, self.best_cost, self.timed_out


def brute_force(fname, coff):
    """
    Entry point matching the original project interface.

    Parameters
    ----------
    fname : str
        Path to the .tsp file (e.g., '../data/Atlanta.tsp').
    coff : int or float
        Cut-off time in seconds.

    Returns
    -------
    current_tour : numpy.ndarray
        Vertex IDs (1-based, as in the .tsp file), with the starting
        vertex repeated at the end to form a closed tour.
    opt_d : float
        Total tour distance, rounded to the nearest integer.
    """
    # Load data using existing helper
    data = read_data(fname)  # list of (vertex_id, x, y)

    solver = BruteForceTSPSolver(data, coff)
    best_idx_tour, best_cost, timed_out = solver.solve(start=0)

    # Map 0-based indices back to vertex IDs from the .tsp file
    vertex_ids = [data[i][0] for i in best_idx_tour]

    # Explicitly close the tour by returning to the start vertex
    if vertex_ids[0] != vertex_ids[-1]:
        vertex_ids.append(vertex_ids[0])

    # Round distance to nearest integer as required
    opt_d = round(best_cost)

    # For compatibility with the original BF.py, return a numpy array
    current_tour = np.array(vertex_ids, dtype=int)

    return current_tour, opt_d