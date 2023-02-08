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
    starting_sequence = str(lines[4])[:-1]
    
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
    
    device = int(lines[13])
    distance_threshold = float(lines[14])
    return wt_sequence, ref_sequence, starting_sequence, metr_mutations, eq_mutations, T, gammas, results_dir, restart_bool, device, distance_threshold


def main(args):
    print('Defining wild-type sequence and simulation parameters...')
    wt_sequence, ref_sequence, starting_sequence, metr_mutations, eq_mutations, T, gammas, results_dir, restart_bool, device, distance_threshold = read_inputs(args.input_file)
    
    print('Initiating algorithm...\n')
    algorithm = Mutation_class(
            wt_sequence = wt_sequence,
            ref_sequence = ref_sequence,
            starting_sequence = starting_sequence,
            metr_mutations = metr_mutations,
            eq_mutations = eq_mutations,
            T = T,
            results_dir = results_dir,
            restart_bool = restart_bool,
            device = device
    )
    
    print('Starting the simulation...')
    gammas = gammas[::-1]
    if len(gammas) > 1:
        algorithm.multimetropolis(gammas)
    else:
        

        algorithm.set_gamma(gammas[0])
        algorithm.set_restart_bool(restart_bool)
        print(f'Simulation gamma: {algorithm.get_gamma()}')
        if eq_mutations > 0:
            print('Equilibration phase...')
            algorithm.metropolis(equilibration = True)
        print('Metropolis...')
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
