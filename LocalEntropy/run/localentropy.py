import numpy as np
import argparse
import os
from os import listdir
from os.path import isfile

from integrator import *


def read_inputs(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    protein_name = str(lines[0])[:-1]
    T = float(lines[1])
    inputs_dir_raw = str(lines[2])[:-1]
    const_step = bool(int(lines[3]))
    return protein_name, T, inputs_dir_raw, const_step

def main(args):
    protein_name, T, inputs_dir_raw, const_step = read_inputs(args.input_file)
    dirlist = [d for d in listdir(f'../RestrainedMetropolis/{inputs_dir_raw}') if isdir(f'../RestrainedMetropolis/{inputs_dir_raw}/{d}') and (f'MM_{protein_name}_T{T}' in d)]
    assert len(dirlist) == 1, 'Too many directories.'
    inputs_dir = f'../RestrainedMetropolis/{inputs_dir_raw}/{dirlist[0]}'
    
    integrator = Integrator(
        inputs_dir = inputs_dir,
        const_step = const_step,
        initialize = True
    )
    '''
    print(f'Protein {protein_type} Local Entropy...')
    print(f'- Simpson method...')
    integrator.Simpson()
    print(f'- Midpoint method...')
    integrator.MidPoint()

    print('Success!')
    '''

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
