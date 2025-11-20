#!/usr/bin/env python3

"""
Executable script.

Run the following command in the terminal:
    - python exec.py -inst <filename> -alg <algorithm> -time <cut-off time> -seed <random seed> 
    - random seed is required only for LS algorithm
    - for example, python exec.py -inst Atlanta.tsp -alg BF -time 3
"""

import argparse
import os
from brute_force import brute_force
from approx import approx
from local_search.local_search import local_search
from plot_tour import plot_tour
from write_output import write_output

if __name__ == "__main__":
    # Obtain user selected arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-inst", required=True, help="filename of dataset")
    parser.add_argument("-alg", required=True, choices=["BF", "Approx", "LS"], help="choose algorithm")
    parser.add_argument("-time", type = int, required=True, help="cut-off time (sec) to terminate algorithm")
    parser.add_argument("-seed", type = int, required=False, default=None, help="random seed (required only for LS)")
    args = parser.parse_args()

    # Formatting for output file
    instance = os.path.basename(args.inst) # filename of dataset
    instance = os.path.splitext(instance)[0] # remove the extension of filename
    method = args.alg # algorithm
    cutoff = args.time # cut-off time (sec)
    random_seed = args.seed # random seed 
    filename = os.path.join("../data", args.inst)

    # Run selected algorithm
    if method == "BF":
        tour, distance = brute_force(filename, cutoff)
    elif method == "Approx":
        tour, distance = approx(filename, cutoff)
    elif method == "LS":
        tour, distance = local_search(filename, cutoff, random_seed)

    print("tour:", tour)
    print("distance:", distance)
    
    # Write output file and plot the tour
    write_output(instance, method, cutoff, tour, distance, random_seed)
    plot_tour(filename, tour, method, cutoff, random_seed)
