from os import listdir
from os.path import isfile, isdir
import numpy as np
import argparse

def load_data(d):
    datafiles = np.array([f for f in listdir(d) if isfile(f'{d}/{f}') and 'data' in f])
    eq_file = datafiles[['eq' in datafile for datafile in datafiles]][0]
    file = datafiles[[not 'eq' in datafile for datafile in datafiles]][0]

    with open(f'{d}/{eq_file}', 'r') as f:
        lines = f.readlines()
    splitted_lines = [line.split('\t') for line in lines]
    eq_data = np.array(splitted_lines).astype(float)

    with open(f'{d}/{file}', 'r') as f:
        lines = f.readlines()
    splitted_lines = [line.split('\t') for line in lines]
    data = np.array(splitted_lines).astype(float)
    return eq_data, data


def load_muts(d):
    mutsfiles = np.array([f for f in listdir(d) if isfile(f'{d}/{f}') and 'mutants' in f])
    eq_file = mutsfiles[['eq' in mutsfile for mutsfile in mutsfiles]][0]
    file = mutsfiles[[not 'eq' in mutsfile for mutsfile in mutsfiles]][0]

    with open(f'{d}/{eq_file}', 'r') as f:
        lines = f.readlines()
    splitted_lines = [line.split('\t') for line in lines]
    eq_muts = np.array(splitted_lines).astype(str)
    for idx in range(len(eq_muts)):
        eq_muts[idx, 1] = eq_muts[idx, 1][:-1]

    with open(f'{d}/{file}', 'r') as f:
        lines = f.readlines()
    splitted_lines = [line.split('\t') for line in lines]
    muts = np.array(splitted_lines).astype(str)
    for idx in range(len(muts)):
        muts[idx, 1] = muts[idx, 1][:-1]

    splitted_d = d.split('_')
    T = float(splitted_d[-2][1:])
    gamma = float(splitted_d[-1][1:])
    return eq_muts, muts, T, gamma

def load_inputs(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    T = lines[0][:-1]
    energy_threshold = float(lines[1])
    section_length = int(lines[2])
    discarded_mutations = int(lines[3])
    d = f'/home/alessandroz/Desktop/LocalEntropy/RestrainedMetropolis/results/SM_1PGB_T{T}_g0.0'
    return T, d, energy_threshold, section_length, discarded_mutations


def main(args):
    print('Loading inputs...')
    T, d, energy_threshold, section_length, discarded_mutations = load_inputs(args.input_file)

    print('Loading mutations and data...')
    _, muts, T, gamma = load_muts(d)
    mutants = muts[discarded_mutations:, 1]
    sections = int(len(mutants) / section_length)
    
    _, data = load_data(d)
    energy = data[discarded_mutations:, 1]

    print('Finding groups...')
    mutants_groups, group = [], []
    dropped = True
    for isec in range(sections):
        energy_mean = energy[isec*section_length:(isec+1)*section_length].mean()
        #print(f'{isec*section_length} - {(isec+1)*section_length}', '\t', energy_mean, '\t', energy_mean < energy_threshold)

        if energy_mean < energy_threshold:
            if dropped:
                print(f'- {len(mutants_groups) + 1}th group found: ', end = '')
            mutants_section = mutants[isec*section_length:(isec+1)*section_length]
            group = group + list(mutants_section)
            dropped = False
        else:
            if len(group) > 0:
                mutants_groups.append(group)
                print(f'{len(mutants_groups[-1])} mutations.')
            dropped = True

        if dropped:
            group = []

    if len(group) > 0: # add last group if not empty
        mutants_groups.append(group)
        print(f'{len(mutants_groups[-1])} mutations.')

    print('Saving groups...')
    for igroup, group in enumerate(mutants_groups):
        with open(f'{args.output_dir}/group_{igroup+1}_T{T}_e{energy_threshold}_dm{discarded_mutations}.dat', 'w') as f:
            for mutant in group:
                print(mutant, file = f)
    
    print('Done!')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type = str,
        default = "inputs/groups_inputs.txt",
        help = "str variable, input file used to load parameters values. Default: 'inputs/groups_inputs.txt'."
    )
    parser.add_argument(
        "--output-dir",
        type = str,
        default = "groups",
        help = "str variable, directory used to save groups sequences. Default: 'groups'."
    )
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
