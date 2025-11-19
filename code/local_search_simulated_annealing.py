#!/usr/bin/env python3

"""
This implements local search (simulated annealing) to solve the Traveling
Salesman Problem.

inputs
    - fname: filename of a dataset (path to .tsp file)
    - coff: time cut-off (sec) to terminate the algorithm
    - rand_seed: random seed for reproducibility

outputs
    - tour: best tour found (numpy array of vertex IDs, start repeated at end)
    - best_cost: total distance of the best tour (rounded to nearest integer)
"""

import time
import math
import random
import numpy as np
from read_data import read_data

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _build_distance_matrix(data):
    """
    Build an integer-rounded Euclidean distance matrix.

    Parameters
    ----------
    data : list of (vertex_id, x, y)

    Returns
    -------
    dist : list of lists
        dist[i][j] is the distance between node i and j (0-based indices).
    """
    n = len(data)
    coords = [(x, y) for (_, x, y) in data]
    dist = [[0] * n for _ in range(n)]

    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            w = int(round(d))  # integer-rounded Euclidean distance
            dist[i][j] = w
            dist[j][i] = w

    return dist


def _tour_cost(perm, dist):
    """
    Compute the cost of a tour given by a permutation of node indices.

    Parameters
    ----------
    perm : list[int]
        Permutation of 0..n-1 (tour is implicitly closed).
    dist : 2D list
        Distance matrix.

    Returns
    -------
    total : int
        Total tour length.
    """
    n = len(perm)
    if n <= 1:
        return 0

    total = 0
    for i in range(n - 1):
        total += dist[perm[i]][perm[i + 1]]
    # close the tour
    total += dist[perm[-1]][perm[0]]
    return total


# ------------------------------------------------------------------------------
# Main simulated annealing routine
# ------------------------------------------------------------------------------

def local_search(fname, coff, rand_seed=0):
    """
    Local search using Simulated Annealing for the TSP.

    Parameters
    ----------
    fname : str
        Path to the .tsp file (e.g., "../DATA/Atlanta.tsp").
    coff : float
        Time cutoff in seconds.
    rand_seed : int, optional
        Random seed for reproducibility (default 0).

    Returns
    -------
    tour : numpy.ndarray
        Best tour found as vertex IDs (1-based), with start vertex repeated at end.
    best_cost : float
        Length of the best tour (rounded to nearest integer).
    """
    # Seed RNGs for reproducibility (ensure Python int, not numpy scalar)
    if rand_seed is None:
        random.seed(None)
        np.random.seed(None)
    else:
        s = int(rand_seed)
        random.seed(s)
        np.random.seed(s)

    # Load data
    data = read_data(fname)  # list of (vertex_id, x, y)
    n = len(data)

    # Handle trivial cases
    if n == 0:
        return np.array([], dtype=int), 0.0
    if n == 1:
        vid = data[0][0]
        return np.array([vid, vid], dtype=int), 0.0

    # Precompute distances
    dist = _build_distance_matrix(data)

    # Initial solution: random permutation of node indices
    current_perm = list(range(n))
    random.shuffle(current_perm)
    current_cost = _tour_cost(current_perm, dist)

    best_perm = list(current_perm)
    best_cost = current_cost

    # --------------------------------------------------------------------------
    # Simulated annealing parameters
    # --------------------------------------------------------------------------

    # Rough scale for initial temperature: average edge length
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append(dist[i][j])
    avg_edge = sum(all_edges) / len(all_edges) if all_edges else 1.0

    T = 10.0 * avg_edge  # initial temperature
    Tmin = 1e-3          # stopping temperature
    alpha = 0.999        # cooling rate (geometric schedule)

    start_time = time.time()

    # --------------------------------------------------------------------------
    # Main annealing loop
    # --------------------------------------------------------------------------
    while T > Tmin:
        # Respect the cutoff
        if time.time() - start_time >= coff:
            break

        # Propose a neighbor: swap two random positions in the permutation
        i, j = random.sample(range(n), 2)
        if i > j:
            i, j = j, i

        neighbor_perm = list(current_perm)
        neighbor_perm[i], neighbor_perm[j] = neighbor_perm[j], neighbor_perm[i]

        neighbor_cost = _tour_cost(neighbor_perm, dist)

        # Metropolis acceptance criterion
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept move
            current_perm = neighbor_perm
            current_cost = neighbor_cost

            # Track the best solution seen so far
            if current_cost < best_cost:
                best_perm = list(current_perm)
                best_cost = current_cost

        # Cool down
        T *= alpha

    # --------------------------------------------------------------------------
    # Map best permutation back to vertex IDs and close the tour
    # --------------------------------------------------------------------------
    vertex_ids = [data[i][0] for i in best_perm]
    if vertex_ids[0] != vertex_ids[-1]:
        vertex_ids.append(vertex_ids[0])

    tour = np.array(vertex_ids, dtype=int)
    best_cost = round(best_cost)

    return tour, best_cost