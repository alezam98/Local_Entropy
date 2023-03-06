import os
import argparse
import numpy as np
from time import time
from tqdm import tqdm


from my_classes import *


def main(args):
    print('Loading experimental sequences...')
    with open(args.input_file, 'r') as f:
        lines = f.readlines()

    exp_mutants = [line.split(' ')[-1][:-1] for line in lines]
    wt_sequence = exp_mutants.pop(0)
    length = len(wt_sequence)

    print('Initiating algorithm...\n')
    algorithm = Mutation_class(
            wt_sequence = wt_sequence,
            ref_sequence = wt_sequence,
            device = args.device
    )
    
    print('Calculating energies and distances...')
    with open(f'{args.input_file[:-4]}_results.txt,', 'w') as f:
        print(f'{wt_sequence}\t0.000\t0.000\t{length}\n', file = f, end = '')

        for exp_mutant in tqdm(exp_mutants):
            distance, _ = algorithm.get_distances(exp_mutant)
            scaled_distance = distance / length
            exp_contacts = algorithm.calculate_contacts(exp_mutant)
            energy = algorithm.calculate_effective_energy(exp_contacts)
            print(f'{exp_mutant}\t{format(scaled_distance, ".3f")}\t{format(energy, ".3f")}\t{length}\n', file = f, end = '')

    print('Done!')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type = str,
        default = "1PGB_alignement.txt",
        help = "str variable, input file used to load parameters values. Default: '1PGB_alignement.txt'."
    )
    parser.add_argument(
        "--device",
        type = int,
        default = 0,
        help = "int variable, device on which the code runs. Default: 0. Choices: 0, 1, 2, 3."
    )
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
