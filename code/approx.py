#!/usr/bin/env python3

"""
This function runs the 2-approximation algorithm for the Traveling Salesman 
Problem.

inputs
    - fname: filename of a dataset (path to .tsp file)
    - coff: time cut-off (sec) to terminate the algorithm (not really needed 
            here, but kept for interface compatibility)

outputs
    - tour: approximate TSP tour (numpy array of vertex IDs, start repeated 
            at end)
    - opt_d: total distance of the tour (float, using integer-rounded 
             Euclidean costs)
"""

import math
import heapq
import numpy as np
from read_data import read_data

# ------------------------------------------------------------------------------
# Distance utilities
# ------------------------------------------------------------------------------

def preorder(T, start):

    tour = [] # initialize the tour
    visited = [] # initialize the visited list

    # run depth-first search (DFS)
    def dfs(u):

        visited.append(u) # add the current vertex to the visited list
        tour.append(u) # add the current vertex to the tour

        n_v = T.neighbors(u) # get the neighbors of the current vertex

        for v in sorted(n_v):
            if v not in visited:
                dfs(v)

    
    dfs(start) # run DFS from the starting vertex
    tour.append(start) # add the starting vertex to the end of the tour

    return tour

def _dist(p1, p2):
    """Integer-rounded Euclidean distance between two (x, y) points."""
    return int(round(math.hypot(p1[0] - p2[0], p1[1] - p2[1])))


def _tour_distance(tour, data):
    """
    Compute total tour length from a tour of vertex IDs.

    tour : sequence of vertex IDs (e.g., [1, 2, 3, 1])
    data : list of (vid, x, y) from read_data
    """
    # Map vertex ID -> (x, y)
    coords = {vid: (x, y) for vid, x, y in data}

    total = 0
    # Sum over edges in the given tour order
    for i in range(len(tour) - 1):
        total += _dist(coords[tour[i]], coords[tour[i + 1]])

    # If the tour is not explicitly closed, add edge back to start
    if len(tour) > 1 and tour[0] != tour[-1]:
        total += _dist(coords[tour[-1]], coords[tour[0]])

    return total


# ------------------------------------------------------------------------------
# MST construction (Prim's algorithm)
# ------------------------------------------------------------------------------

class MSTree:
    """Simple tree wrapper with a .neighbors(u) method for preorder()."""

    def __init__(self, adj):
        # adj: dict[vertex_id] -> list[neighbor_vertex_id]
        self.adj = adj

    def neighbors(self, u):
        return self.adj[u]


def _build_mst(data):
    """
    Build a Minimum Spanning Tree using Prim's algorithm.

    data: list of (vertex_id, x, y)

    Returns
    -------
    T : MSTree
        Tree with .neighbors(u) -> iterable of neighbor vertex IDs.
    """
    n = len(data)
    vids = [v for (v, x, y) in data]
    coords = [(x, y) for (v, x, y) in data]

    INF = float("inf")
    in_mst = [False] * n
    key = [INF] * n
    parent = [-1] * n

    # Start MST from the first vertex (index 0)
    key[0] = 0
    pq = [(0, 0)]  # (key, vertex_index)

    while pq:
        w, u = heapq.heappop(pq)
        if in_mst[u]:
            continue
        in_mst[u] = True

        x1, y1 = coords[u]
        # Relax edges to all other vertices (complete graph)
        for v in range(n):
            if not in_mst[v]:
                x2, y2 = coords[v]
                d = _dist((x1, y1), (x2, y2))
                if d < key[v]:
                    key[v] = d
                    parent[v] = u
                    heapq.heappush(pq, (d, v))

    # Build adjacency list for the MST in terms of vertex IDs
    adj = {vid: [] for vid in vids}
    for v in range(1, n):
        p = parent[v]
        if p == -1:
            continue
        parent_vid = vids[p]
        child_vid = vids[v]
        adj[parent_vid].append(child_vid)
        adj[child_vid].append(parent_vid)

    return MSTree(adj)


# ------------------------------------------------------------------------------
# Main Approx entry point
# ------------------------------------------------------------------------------

def Approx(fname, coff):
    """
    2-approximation TSP via MST + preorder traversal.

    Parameters
    ----------
    fname : str
        Path to .tsp file (e.g., "../DATA/Atlanta.tsp").
    coff : float
        Time cutoff in seconds (not used for early stopping here,
        but kept for interface compatibility).

    Returns
    -------
    tour : numpy.ndarray
        Vertex IDs forming a tour, with the starting vertex repeated at the end.
    opt_d : float
        Total tour length using integer-rounded Euclidean distances.
    """
    # Load data: list of (vertex_id, x, y)
    data = read_data(fname)
    if len(data) == 0:
        return np.array([], dtype=int), 0.0

    # Build MST
    mst_tree = _build_mst(data)

    # Choose the first vertex ID as the start
    start_vid = data[0][0]

    # Preorder traversal of the MST gives the "double-tree" shortcut tour
    tour_list = preorder(mst_tree, start_vid)  # e.g., [1, 2, 3, ..., 1]

    # Compute total distance
    opt_d = float(_tour_distance(tour_list, data))

    # Return as numpy array for consistency with BF
    tour = np.array(tour_list, dtype=int)

    return tour, opt_d