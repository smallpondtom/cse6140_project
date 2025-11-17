#!/usr/bin/env python3

"""
This is the test script to run the brute force (BF) algorithm

Run the following command in the terminal:
  - python test_brute_force.py -inst <filename> -alg BF -time 600
  - put the filename of a dataset in <filename>
  - for example, python test_bruce_force.py -inst Atlanta.tsp -alg BF -time 10
"""

import argparse
import os
from brute_force import brute_force
from plot_tour import plot_tour
from write_output import write_output

if __name__ == "__main__":
    # Obtain arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-inst", required=True, help="filename of a dataset")
    parser.add_argument("-alg", required=True, choices=["BF", "Approx", "LS"], help="choose the algorithm")
    parser.add_argument("-time", type = int, required=True, help="cut-off time (sec) to terminate the algorithm")
    parser.add_argument("-seed", type = int, required=False, help="random seed (required only for LS)")
    args = parser.parse_args()

    # Formatting the output file
    instance = os.path.basename(args.inst)   # filename of a dataset
    instance = os.path.splitext(instance)[0] # remove the extension of the filename
    method = args.alg                        # algorithm
    cutoff = args.time                       # cut-off time (sec)
    random_seed = args.seed                  # random seed
    filename = os.path.join("../data", args.inst)

    # Run brute-force algorithm
    tour, distance = brute_force(filename, cutoff)

    print("tour:", tour)
    print("distance:", distance)
    
    # Write the output file and plot the tour
    write_output(instance, method, cutoff, tour, distance, None)
    plot_tour(filename, tour, method, cutoff, None)
