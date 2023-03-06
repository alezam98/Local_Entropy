import os
import argparse
import numpy as np
from time import time

from my_classes import *


def read_inputs(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    simtype = str(lines[0])[:-1]
    assert simtype in ('SM', 'MM'), 'Wrong simulation type input.' 

    protein_name = str(lines[1])[:-1]
    wt_sequence = str(lines[2])[:-1]
    ref_sequence = str(lines[3])[:-1]
    if ref_sequence == '': ref_sequence = wt_sequence
    starting_sequence = str(lines[4])[:-1]
    if starting_sequence == '': starting_sequence = ref_sequence
    
    metr_mutations = int(lines[5])
    eq_mutations = int(lines[6])
    T = float(lines[7])
    gammas = np.linspace(float(lines[8]), float(lines[9]), int(lines[10]))
    if simtype == 'SM':
        assert len(gammas) == 1, 'Too many gamma according to the simulation type.'
    
    results_dir = lines[11][:-1]
    if simtype == 'SM' and len(gammas) == 1:
        results_dir = f'{results_dir}/{simtype}_{protein_name}_T{str(T)}_g{str(gammas[0])}'
    elif simtype == 'MM' and len(gammas) > 1:
        results_dir = f'{results_dir}/{simtype}_{protein_name}_T{str(T)}'
    restart_bool = bool(int(lines[12]))
    unique_length = int(lines[13])
    
    device = int(lines[14])
    distance_threshold = float(lines[15])
    return simtype, wt_sequence, ref_sequence, starting_sequence, metr_mutations, eq_mutations, T, gammas, results_dir, restart_bool, unique_length, device, distance_threshold


def main(args):
    print('Defining wild-type sequence and simulation parameters...')
    simtype, wt_sequence, ref_sequence, starting_sequence, metr_mutations, eq_mutations, T, gammas, results_dir, restart_bool, unique_length, device, distance_threshold = read_inputs(args.input_file)
    
    print('Initiating algorithm...\n')
    if simtype == 'MM' and restart_bool:        
        restart_bool = False
        print('Restart option is not available for MM simulations. Setting restart_bool = False.')

    algorithm = Mutation_class(
            wt_sequence = wt_sequence,
            ref_sequence = ref_sequence,
            starting_sequence = starting_sequence,
            metr_mutations = metr_mutations,
            eq_mutations = eq_mutations,
            T = T,
            results_dir = results_dir,
            restart_bool = restart_bool,
            unique_length = unique_length,
            device = device
    )
    
    print('Starting the simulation...')
    gammas = gammas[::-1]
    for igamma, gamma in enumerate(gammas):
        algorithm.set_gamma(gamma)
        algorithm.set_starting_sequence(starting_sequence)
        algorithm.set_restart_bool(restart_bool)
        
        print(f'- Simulation gamma: {algorithm.get_gamma()}')
        if igamma == 0 and eq_mutations > 0:
            print(f'  Equilibration phase...')
            algorithm.metropolis(equilibration = True)
        print('  Metropolis...')
        algorithm.metropolis()
        
        starting_sequence = algorithm.get_last_sequence()
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
