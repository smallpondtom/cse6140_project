#!/usr/bin/env python3

"""
Write the output file for TSP solutions as .sol format.

The output file contains two lines:
    - first line: the total distance of the optimal tour (floating point)
    - second line: the optimal tour (list of integers, comma-separated)

# Input
- instance: filename of a dataset
- method: the algorithm
- cutoff: cut-off time (sec) to terminate the algorithm
- seed: random seed (required only for LS)
- tour: optimal tour found by the algorithm
- distance: total distance of the optimal tour

# Output
- output file .sol
"""
def write_output(instance, method, cutoff, tour, distance, seed=None):
    # Format the name of the output file
    if method == "LS":
        out = f"../output/{instance}_{method}_{cutoff}_{seed}.sol"
    else:
        out = f"../output/{instance}_{method}_{cutoff}.sol"

    distance = float(distance) 
    tour = list(map(int, tour)) 
    tour = map(str, tour) 

    with open(out, "w") as file:
        file.write(f"{distance}\n")
        file.write(','.join(tour))