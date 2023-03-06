import numpy as np
import pandas as pd
import argparse
from os import listdir
from os.path import isfile, isdir

from distribution_class import *


def load_inputs(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    T = float(lines[0])
    energy_threshold = float(lines[1])
    dtype = str(lines[2])[:-1]
    step = int(lines[3])
    prec = float(lines[4])
    discarded_mutations = int(lines[5])
    inputs_d = str(lines[6])[:-1]
    groups_d = str(lines[7])[:-1]
    pdfs_d = str(lines[8])[:-1]
    restart = bool(int(lines[9]))

    return T, energy_threshold, dtype, step, prec, discarded_mutations, inputs_d, groups_d, pdfs_d, restart


def main(args):
    print(f'pid: {os.getpid()}')

    print('Loading inputs...')
    T, energy_threshold, dtype, step, prec, discarded_mutations, inputs_d, groups_d, pdfs_d, restart = load_inputs(args.input_file)

    print('Preparing algorithm...')
    algorithm = Distribution_class(
            T = T,
            energy_threshold = energy_threshold,
            dtype = dtype,
            step = step,
            prec = prec,
            discarded_mutations = discarded_mutations,
            inputs_d = inputs_d,
            groups_d = groups_d,
            pdfs_d = pdfs_d,
            restart = restart
    )

    print('\nStarting the calculation for same pdfs...')
    algorithm.get_group_distribution(pdftype = 'same')

    print('\nStarting the calculation for mutual pdfs...')
    algorithm.get_group_distribution(pdftype = 'mutual')

    print('\nDone!')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type = str,
        default = "inputs/main_inputs.txt",
        help = "str variable, input file used to load parameters values. Default: 'inputs/main_inputs.txt'."
    )
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
