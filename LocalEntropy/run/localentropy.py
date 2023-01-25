#!/usr/bin/env python
import numpy as np
import argparse
import os
from os import listdir
from os.path import isfile

from integrator.py import *


def read_inputs(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    discarded_mutations = int(lines[0])
    inputs_dir_raw = str(lines[1])[:-1]
    const_step = bool(int(lines[2]))
    return discarded_mutations, inputs_dir_raw, const_step

def main(args):
    discarded_mutations, inputs_dir_raw, const_step = read_inputs(args.input_file)
    seq_types = [dir_ for dir_ in listdir(f'../RestrainedMetropolis/results/{inputs_dir_raw}') if isdir(f'../RestrainedMetropolis/results/{inputs_dir_raw}/{dir_}')]

    integrator = Integrator(
        discarded_mutations = discarded_mutations,
        const_step = const_step,
        initialize = False
    )

    print(f'Protein {inputs_dir_raw}')
    for idx, seq_type in enumerate(seq_types):
        print(f'- sequence: {seq_type}\t({idx}/{len(seq_types)})')
        inputs_dir = inputs_dir_raw + '/' + seq_type

        print(f'  loading data and calculating means...')
        integrator.set_inputs_dir(inputs_dir)
        integrator.initialize()

        print(f'  Simpson method...')
        integrator.Simpson()
        print(f'  Midpoint method...')
        integrator.MidPoint()

        print()

    print('Success!')


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
