#!/usr/bin/env python
import glob
import argparse

'''Discard incomplete data'''
def main(args):
    datapath = args.results_dir + '/data_*'
    datalist = glob.glob(datapath)

    print('Start the cleaning...')
    for idx, datafile in enumerate(datalist):
        print(f'{idx + 1}/{len(datalist)}... ', end = '')

        dir_, file_id = datafile.split('data')
        mutsfile = f'{dir_}mutants{file_id}'
        
        with open(mutsfile, 'r') as file:
            muts_lines = file.readlines()
            muts_generation = len(muts_lines) - 1   # don't count the wt sequence
        with open(datafile, 'r') as file:
            data_lines = file.readlines()
            data_generation = len(data_lines)
            
        if muts_generation < data_generation:
            generation = muts_generation
            with open(datafile, 'w') as file:
                for line in data_lines[:generation]:
                    print(line, end = '', file = file)
        elif muts_generation > data_generation:
            generation = data_generation
            with open(mutsfile, 'w') as file:
                for line in muts_lines[:generation + 1]:
                    print(line, end = '', file = file)

        print('done.')
    print('Success.')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type = str,
        default = "./results",
        help = "str variable, directory containing saved data and mutations. Default: './results'."
    )
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
