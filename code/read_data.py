#!/usr/bin/env python3
"""Read TSP data from a .tsp file."""

def read_data(filename):
    data = []
    # read data from the file
    readstart = False # set flag to indicate where to read from
    with open(filename, 'r') as f:
        for line in f:
            # we start reading data from the line below of NODE_COORD_SECTION
            if line == "NODE_COORD_SECTION\n":
                readstart = True
                continue
            if readstart:
                if line == "EOF\n":
                    break
                out = line.split()
                vid = int(out[0]) # save vertex ID
                x = float(out[1]) # save x coords
                y = float(out[2]) # save y coords
                data.append((vid, x, y))   
    return data
