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
    ref_sequence = str(lines[1])[:-1]
    if ref_sequence == '': ref_sequence = wt_sequence
    metr_mutations = int(lines[2])
    eq_mutations = int(lines[3])
    threshold = float(lines[4])
    T = float(lines[5])
    gammas = np.linspace(float(lines[6]), float(lines[7]), int(lines[8]))
    results_dir = lines[9][:-1]
    restart_bool = bool(int(lines[10]))
    device = int(lines[11])
    distance_threshold = float(lines[12])
    return wt_sequence, ref_sequence, metr_mutations, eq_mutations, threshold, T, gammas, results_dir, restart_bool, device, distance_threshold


def main(args):
    print('Defining wild-type sequence and simulation parameters...')
    wt_sequence, ref_sequence, metr_mutations, eq_mutations, threshold, T, gammas, results_dir, restart_bool, device, distance_threshold = read_inputs(args.input_file)
    
    print('Initiating algorithm...\n')
    algorithm = Mutation_class(
            wt_sequence = wt_sequence,
            ref_sequence = ref_sequence,
            metr_mutations = metr_mutations,
            eq_mutations = eq_mutations,
            threshold = threshold,
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
