#!/usr/bin/env python
import numpy as np
from os import listdir
from os.path import isfile, isdir


class Integrator:

    ### Initialization
    def __init__(
        self,
        discarded_mutations : int,
        inputs_dir : str,
        const_step : bool = True,
        initialize : bool = False
    ):
        assert discarded_mutations >= 0, "discarded_mutations can't be negative."
        self.discarded_mutations = discarded_mutations
        self.const_step = const_step

        if inputs_dir[-1] == '/': inputs_dir = inputs_dir[:-1]
        inputs_dir = inputs_dir.split('/')[-2:]
        self.inputs_dir = f'../RestrainedMetropolis/results/{inputs_dir[0]}/{inputs_dir[1]}'
        self.results_dir = f'../results/{inputs_dir[0]}'
        self._get_id(inputs_dir[1])
        self._check_dir()

        if initialize: self.initialize()



    ### Prepare data file id
    def _get_id(self, protein_type):
        filelist = [filename for filename in listdir(f'{self.inputs_dir}') if isfile(f'{self.inputs_dir}/{filename}')]
        file_ids = [filename.split('_')[1:3] for filename in filelist]
        file_id = np.unique(file_ids)
        assert len(file_id) == 1, f"Different simulation parameters in {self.inputs_dir} directory."
        self.file_id = f'{protein_type}_{file_id[0]}'



    ### Check for directory to store integration results (derived from inputs_dir)
    def _check_directory(self):
        path = self.results_dir.split('/')[1:]
        actual_dir = '..'
        for idx, new_dir in enumerate(path):
            if idx > 0:
                actual_dir = actual_dir + '/' + path[idx - 1]
            onlydirs = [d for d in listdir(f'{actual_dir}') if isdir(f'{actual_dir}/{d}')]
            if (new_dir in onlydirs) == False:
                os.mkdir(f'{actual_dir}/{new_dir}')



    ### Prepare for integration
    def initialize(self):
        self.load_data(self.discarded_mutations, self.const_step)
        self.calculate_means_matrix()



    ### Load data from inputs_dir
    def load_data(self, discarded_mutations, const_step):
        # Load dlists and gamma
        gammas, dlists = [], []
        data_files = [filename for filename in listdir(f'{self.inputs_dir}') if isfile(f'{self.inputs_dir}/{filename}') and ('data' in filename)]

        for data_file in data_files:
            with open(f'{self.inputs_dir}/{data_file}', 'r') as file:
                lines = file.readlines()
            gamma = float(lines[0].split('\t')[-2])
            dlist = [float(line.split('\t')[3]) for line in lines]
            gammas.append(gamma)
            dlists.append(dlist[discarded_mutations:])

        self.gammas = np.sort(gammas)[::-1]
        self.dlists = np.array([dlist[gammas.index(gamma)] for gamma in ord_gammas], dtype = float)[::-1]

        if const_step:
            shifted_gammas = np.append(self.gammas[1:], [self.gammas[0]])
            steps = (self.gammas - shifted_gammas)[:-1]
            assert len(np.unique(steps)) == 1, "Different intervals between simulation gammas."

        # Load S0 value
        min_idx = np.argmin(gammas)
        with open(f'{self.inputs_dir}/{data_files[min_idx]}', 'r') as file:
            lines = file.readlines()
        self.U = float(lines[0].split('\t')[1])
        self.beta = float(lines[0].split('\t')[-3])
        self.S0 = -self.beta * self.U



    ### Calculate means_matrix
    def calculate_means_matrix(self):
        self.means_matrix = np.zeros((len(self.gammas), len(self.gammas)), dtype = float)
        for row in range(len(self.gammas)):
            self.means_matrix[row, row] = np.mean(self.dlists[row])
            for col in range(len(self.gammas)):
                if col == row: continue
                self.means_matrix[row, col] = self.reweighted_mean(self.dlists[col], self.gammas[col], self.gammas[row])
        self.means = np.mean(means_matrix, axis = 1)



    ### Reweight distances list
    def reweighted_mean(self, dlist, old_gamma, new_gamma):
        dlist = np.array(dlist)
        denominator = np.exp(-(new_gamma - old_gamma) * dlist)
        numerator = dlist * denominator
        return np.mean(numerator) / np.mean(denominator)



    ### Compute mean through reweighting
    def predict_mean(self, new_gamma):
        mean = 0.
        for dlist, old_gamma in zip(self.dlists, self.gammas):
            mean += self.reweighted_mean(dlist, old_gamma, new_gamma)
        return mean / len(self.gammas)



    ### Simpson method
    def Simpson(self):
        self.S = np.zeros(len(self.gammas), dtype = float)
        self.S[0] = self.S0

        for idx in range(len(self.gammas[:-1])):
            step = (self.gammas[idx + 1] - self.gammas[idx])/2.
            predicted_mean = self.predict_mean(self.gammas[idx] + step)
            increment = (step / 6.) * (self.means[idx] + 4.*predicted_mean + self.means[idx + 1])
            self.S[idx + 1] = self.S[idx] + increment

        with open(f'{self.results_dir}/Simpson_{self.file_id}.dat', 'w') as file:
            for gamma_n, S_n, dS_n in zip(self.gammas, self.S, self.means):
                print(f'{gamma_n}\t{S_n}\t{dS_n}', file = file)



    ### MidPoint integration method
    def MidPoint(self):
        self.S = np.zeros(len(self.gammas), dtype = float)
        self.S[0] = self.S0

        for idx in range(len(self.gammas[:-1])):
            step = (self.gammas[idx + 1] - self.gammas[idx])/2.
            increment = (step / 2.) * (self.means[idx] + self.means[idx + 1])
            self.S[idx + 1] = self.S[idx] + increment

        with open(f'{self.results_dir}/MidPoint_{self.file_id}.dat', 'w') as file:
            for gamma_n, S_n, dS_n in zip(self.gammas, self.S, self.means):
                print(f'{gamma_n}\t{S_n}\t{dS_n}', file = file)



    ### Set modules
    def set_discarded_mutations(self, discarded_mutations : int):
        assert discarded_mutations >= 0, "discarded_mutations can't be negative."
        self.discarded_mutations = discarded_mutations
        self.load_data(self.discarded_mutations, self.const_step)

    def set_inputs_dir(self, inputs_dir : str):
        if inputs_dir[-1] == '/': inputs_dir = inputs_dir[:-1]
        inputs_dir = inputs_dir.split('/')[-2:]
        self.inputs_dir = f'../RestrainedMetropolis/results/{inputs_dir[0]}/{inputs_dir[1]}'
        self.results_dir = f'../results/{inputs_dir[0]}'
        self._get_id(inputs_dir[1])
        self._check_dir()

    def set_const_step(self, const_step : bool):
        self.const_step = const_step
        self.load_data(self.discarded_mutations, self.const_step)



    ### Get modules
    def get_discared_mutations(self): return self.discarded_mutations
    def get_const_step(self): return self.const_step
    def get_inputs_dir(self): return self.inputs_dir
    def get_results_dir(self): return self.results_dir
    def get_file_id(self): return self.file_id
    def get_beta(self): return self.beta
    def get_U(self): return self.U
    def get_S0(self): return self.S0
    def get_gammas(self): return self.gammas
    def get_dlists(self): return self.dlists
    def get_means_matrix(self): return self.means_matrix
    def get_means(self): return self.means
    def get_S(self): return self.S
