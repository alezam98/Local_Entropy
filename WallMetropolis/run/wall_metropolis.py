import os
import argparse
import numpy as np
from time import time

from my_wall_classes import *


def read_inputs(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    protein_name = str(lines[0])[:-1]
    wt_sequence = str(lines[1])[:-1]
    ref_sequence = str(lines[2])[:-1]
    if ref_sequence == '': ref_sequence = wt_sequence
    starting_sequence = str(lines[3])[:-1]
    if starting_sequence == '': starting_sequence = ref_sequence
    
    metr_mutations = int(lines[4])
    eq_mutations = int(lines[5])
    T = float(lines[6])
    dmax = float(lines[7])
    results_dir = lines[8][:-1]
    results_dir = f'{results_dir}/{protein_name}_T{str(T)}_d{str(dmax)}'
    restart_bool = bool(int(lines[9]))
    unique_length = int(lines[10])
    
    device = int(lines[11])
    distance_threshold = float(lines[12])
    return wt_sequence, ref_sequence, starting_sequence, metr_mutations, eq_mutations, T, dmax, results_dir, restart_bool, unique_length, device, distance_threshold


def main(args):
    print('Defining wild-type sequence and simulation parameters...')
    wt_sequence, ref_sequence, starting_sequence, metr_mutations, eq_mutations, T, dmax, results_dir, restart_bool, unique_length, device, distance_threshold = read_inputs(args.input_file)
    
    print('Initiating algorithm...\n')
    algorithm = Mutation_class(
            wt_sequence = wt_sequence,
            ref_sequence = ref_sequence,
            starting_sequence = starting_sequence,
            metr_mutations = metr_mutations,
            eq_mutations = eq_mutations,
            T = T,
            dmax = dmax,
            results_dir = results_dir,
            restart_bool = restart_bool,
            unique_length = unique_length,
            device = device
    )
    
    print('Starting the simulation...')
    print(f'- Simulation dmax: {algorithm.get_dmax()}')
    if eq_mutations > 0:
        print(f'  Equilibration phase...')
        algorithm.metropolis(equilibration = True)
    print('  Metropolis...')
    algorithm.metropolis()
        
    print('Done!')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type = str,
        default = "inputs/inputs.txt",
        help = "str variable, input file used to load parameters values. Default: 'inputs/inputs.txt'."
    )
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
