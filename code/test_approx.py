#!/usr/bin/env python3

"""
This is the test script to run the 2-approximation algorithm.

Run the following in the terminal:
    - python test_approx.py -inst <filename> -alg Approx -time 600
    - put the filename of a dataset in <filename>
    - for example, python test_approx.py -inst Atlanta.tsp -alg Approx -time 600
"""

import argparse
import os
from approx import Approx
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
    instance = os.path.basename(args.inst) # get the filename of a dataset
    instance = os.path.splitext(instance)[0] # remove the extension of the filename
    method = args.alg # get the algorithm
    cutoff = args.time # get the cut-off time
    random_seed = args.seed # get the random seed

    filename = os.path.join("../data", args.inst)

    # Run 2-approximation algorithm
    tour, distance = Approx(filename, cutoff)

    print("tour:", tour)
    print("distance:", distance)

    # Write the output file and plot the tour
    write_output(instance, method, cutoff, tour, distance, None)
    plot_tour(filename, tour, method, cutoff, None)
