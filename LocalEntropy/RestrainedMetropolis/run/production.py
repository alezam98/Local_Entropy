#!/usr/bin/env python
import os
import argparse
import numpy as np
#import time

from my_new_classes import *


def read_inputs(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    wt_sequence = str(lines[0])[:-1]
    num_mutations = int(lines[1])
    sensitivity = float(lines[2])
    T = float(lines[3])
    gammas = np.linspace(float(lines[4]), float(lines[5]), int(lines[6]))
    results_dir = lines[7][:-1]
    restart_bool = bool(int(lines[8]))
    device = int(lines[9])
    distance_threshold = float(lines[10])
    return wt_sequence, num_mutations, sensitivity, T, gammas, results_dir, restart_bool, device, distance_threshold


def main(args):
    print('Defining wild-type sequence and simulation parameters...')
    wt_sequence, num_mutations, sensitivity, T, gammas, results_dir, restart_bool, device, distance_threshold = read_inputs(args.input_file)
    
    print('Initiating algorithm...\n')
    algorithm = Mutation_class(
            wt_sequence = wt_sequence,
            num_mutations = num_mutations,
            sensitivity = sensitivity,
            T = T,
            results_dir = results_dir,
            restart_bool = restart_bool,
            device = device
    )
    
    print('Starting the simulation...')
    algorithm.multimetropolis(gammas)
    print('Done!')

    

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type = str,
        default = "inputs.txt",
        help = "str variable, input file used to load parameters values. Default: 'inputs.txt'."
    )
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
