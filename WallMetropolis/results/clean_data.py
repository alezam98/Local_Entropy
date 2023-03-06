from os import listdir
from os.path import isdir, isfile
import glob
import argparse

'''Discard incomplete data'''
def main(args):
    dirlist = [d for d in listdir(args.results_dir) if isdir(f'{args.results_dir}/{d}')]
    print('Start the cleaning...\n')
    for d in dirlist:
        print(f'- {d} directory')
        curr_dir = f'{args.results_dir}/{d}'
        datapath = f'{curr_dir}/data_*'
        datalist = glob.glob(datapath)

        for idx, datafile in enumerate(datalist):
            print(f'  {idx + 1}/{len(datalist)}... ', end = '')

            dir_, file_id = datafile.split('data')
            mutsfile = f'{dir_}mutants{file_id}'
        
            with open(mutsfile, 'r') as file:
                muts_lines = file.readlines()
                muts_num = len(muts_lines)
            with open(datafile, 'r') as file:
                data_lines = file.readlines()
                data_num = len(data_lines)
            
            if muts_num < data_num:
                with open(datafile, 'w') as file:
                    for line in data_lines[:muts_num]:
                        print(line, end = '', file = file)
            elif muts_num > data_num:
                with open(mutsfile, 'w') as file:
                    for line in muts_lines[:data_num]:
                        print(line, end = '', file = file)
            print('done.\n')
    print('Success.')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type = str,
        default = ".",
        help = "str variable, directory containing saved data and mutations. Default: '.'."
    )
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
