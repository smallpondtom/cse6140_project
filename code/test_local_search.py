"""
This is the test script to run the local search (LS) algorithm.

Run the following command in the terminal:
    - python test_local_search.py -inst <filename> -alg LS -time 10 -seed 1
    - put the filename of a dataset in <filename>
    - for example, python test_local_search.py -inst Atlanta.tsp -alg LS -time 10 -seed 1
"""

import argparse
import os
#from local_search.local_search import local_search
from local_search_simulated_annealing import local_search
from plot_tour import plot_tour
from write_output import write_output

if __name__ == "__main__":
    # Obtain arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-inst", required=True, help="filename of dataset")
    parser.add_argument("-alg", required=True, choices=["BF", "Approx", "LS"], help="choose algorithm")
    parser.add_argument("-time", type = int, required=True, help="cut-off time (sec) to terminate algorithm")
    parser.add_argument("-seed", type = int, required=False, default=1, help="random seed (required only for LS)")
    args = parser.parse_args()

    # Formatting the output file
    instance = os.path.basename(args.inst) # filename of dataset
    instance = os.path.splitext(instance)[0] # remove the extension of the filename
    method = args.alg # algorithm
    cutoff = args.time # cut-off time (sec)
    random_seed = args.seed # random seed
    filename = os.path.join("../data", args.inst)

    # Run LS algorithm
    tour, distance = local_search(filename, cutoff, random_seed)

    print("tour:", tour)
    print("distance:", distance)

    # Write the output file and plot the tour
    write_output(instance, method, cutoff, tour, distance, random_seed)
    plot_tour(filename, tour, method, cutoff, random_seed)
