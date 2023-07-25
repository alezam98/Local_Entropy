import os
import argparse
import numpy as np
from time import time
from os import listdir
from os.path import isdir
from subprocess import call

from utils import load_inputs
from my_replica_classes import *


def check_parameters(parameters):
    # Define gamma values
    parameters['gammas'] = np.linspace(parameters['gamma_i'], parameters['gamma_f'], parameters['num_gammas'])

    # Check for <protein_name> directory
    first_dir = f"{parameters['results_dir']}/{parameters['protein_name']}"
    directories = [f"{parameters['results_dir']}/{d}" for d in listdir(f"../{parameters['results_dir']}") if isdir(f"../{parameters['results_dir']}/{d}")]
    if not first_dir in directories:
        # Create <protein_name> directory
        command = f"mkdir ../{first_dir}".split(' ')
        call(command)
        # Save wild-type sequence and length
        with open(f"../{first_dir}/README.txt", 'w') as f:
            print(f"Protein name:     {parameters['protein_name']}", file = f)
            print(f"Protein sequence: {parameters['wt_sequence']}", file = f)
            print(f"Protein length:   {len(parameters['wt_sequence'])}", file = f)
    
    # Define results directory
    if not parameters['load']: 
        second_dir = "NE"
        third_dir = f"y{parameters['y']}_s{parameters['seed']}_e{parameters['eq_energy']}"
    else: 
        second_dir = "AE"
        third_dir = f"y{parameters['y']}_s{parameters['seed']}_c{parameters['comb']}"
    if len(parameters['gammas']) == 1: fourth_dir = f"T{parameters['T']}_g{parameters['gamma_i']}"
    else: fourth_dir = f"T{parameters['T']}"
    parameters['results_dir'] = f"{first_dir}/{second_dir}/{third_dir}/{fourth_dir}"

    return parameters


def main(args):
    print(f'PID: {os.getpid()}\n')

    print('Defining wild-type sequence and simulation parameters...')
    tuples = [
        ('protein_name', str),
        ('wt_sequence', str),
        ('mutations', int),
        ('T', float),
        ('gamma_i', float),
        ('gamma_f', float),
        ('num_gammas', int),
        ('y', int),
        ('seed', int),
        ('eq_energy', float),
        ('unique_length', int),
        ('results_dir', str),
        ('step', int),
        ('restart', bool),
        ('load', bool),
        ('comb', int),
        ('device', int),
        ('distance_threshold', float)
    ]
    parameters = load_inputs(args.input_file, tuples)
    parameters = check_parameters(parameters)

    print('Initiating algorithm...\n')
    eq_mutations, eq_gamma = 1000, 0.
    algorithm = Mutation_class(
            wt_sequence = parameters['wt_sequence'],
            mutations = eq_mutations,
            T = parameters['T'],
            gamma = eq_gamma,
            y = parameters['y'],
            seed = parameters['seed'],
            unique_length = parameters['unique_length'],
            results_dir = parameters['results_dir'],
            step = parameters['step'],
            restart = parameters['restart'],
            load = parameters['load'],
            comb = parameters['comb'],
            device = parameters['device'],
            distance_threshold = parameters['distance_threshold']
    )
    
    if not parameters['load']:
        alogrithm.print_status()
        print(''.join(['-'] * 100))
        print(f'Equilibration phase, indipendent evolution. Request: <E> < {parameters["eq_energy"]}')
        mean_energy = algorithm.replica_energies.mean()
        while mean_energy >= parameters['eq_energy']:
            algorithm.print_progress()
            algorithm.metropolis(save = True, print_progress = False)
            mean_energy = algorithm.replica_energies.mean()
        print('The system is now below the equilibration energy.\n')

    print(''.join(['-'] * 100))
    print('Starting the simulation...')
    algorithm.set_mutations(parameters['mutations'])
    for gamma in parameters['gammas']:
        algorithm.set_gamma(gamma)
        algorithm.print_status()
        algorithm.metropolis(save = True, print_progress = True)
    print('Simulation completed!')
    

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
