#!/usr/bin/env python3
"""Plot the TSP tour."""

import matplotlib.pyplot as plt
import numpy as np
import os
from read_data import read_data

def plot_tour(filename, tour, method, cutoff, seed=None):
    # Load data
    data = read_data(filename)
    data = np.array(data)
    coords = data[:,1:]

    tour_id = [int(id)-1 for id in tour]
    tour_coords = coords[tour_id]

    plt.figure(figsize=(10, 6))
    plt.plot(tour_coords[:,0], tour_coords[:,1], 'o-', label='tour')

    for i, (x,y) in enumerate(tour_coords):
        plt.text(x, y, str(i), fontsize=12)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    # Save the plot
    filename = os.path.splitext(os.path.basename(filename))[0]
    if method == "LS":
        out = f"../plot/{filename}_{method}_{cutoff}_{seed}png"
    else:
        out = f"../plot/{filename}_{method}_{cutoff}.png"
    plt.savefig(out)
