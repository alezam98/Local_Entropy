#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.special import softmax
from jax.tree_util import tree_map
#from tqdm import tqdm
import os, glob, sys
from subprocess import call
from os import listdir
from os.path import isfile, isdir
#import time

import torch
import esm

#from prediction_class import *

seed = 0
np.random.seed(seed)


### -------------------------------------- BASIC ALGORITHM ------------------------------------- ###
class Basic_class:

    ### Initialization
    def __init__(
            self,
            device : int = 0,
            distance_threshold : float = 4.
    ):

        torch.cuda.empty_cache()
    
        self.device = device
        torch.cuda.set_device(device)

        self.model = esm.pretrained.esmfold_v1()
        self.model = self.model.eval().cuda()

        self.distance_threshold = distance_threshold

        print('Basic class, status:')
        print(f'model: esmfold_v1')
        print(f'device: {self.device}')
        print(f'distance threshold: {self.distance_threshold} [A]', '\n')


    ### Calculate contact map through esmfold model
    def calculate_contacts(self, sequence, method = 'explicit', return_plddt = False):
        with torch.no_grad():
            output = self.model.infer(sequence)
        output = tree_map(lambda x: x.cpu().numpy(), output)

        plddt = output['plddt'][0, :, 1]
        plddt_mask = output['atom37_atom_exists'][0].astype(bool)        
        self._check_Ca(plddt_mask, sequence)

        if method == 'explicit':
            positions = output['positions'][-1, 0]
            positions_mask = output['atom14_atom_exists'][0].astype(bool)
            distance_matrix = self._calculate_distance_matrix(positions, positions_mask)
            contact_map = (distance_matrix < self.distance_threshold).astype(int)

        elif method == 'implicit':
            bins = np.append(0, np.linspace(2.3125, 21.6875, 63))
            contact_map = softmax(output['distogram_logits'], -1)[0]
            contact_map = contact_map[..., bins < 8].sum(-1)
            
        torch.cuda.empty_cache()
        del output
        
        if return_plddt:
            return contact_map, plddt
        else:
            return contact_map



    ### Calculate distance matrix for given chain
    def _calculate_distance_matrix(self, positions, positions_mask):
        distance_matrix = np.zeros( (len(positions), len(positions)) )
        idxs = np.arange(0, len(positions))
        for row in idxs:
            for col in idxs[idxs > row]:
                residue_one = positions[row, positions_mask[row]]
                residue_two = positions[col, positions_mask[col]]
                distance_matrix[row, col] = self._calculate_residue_distance(residue_one, residue_two)
                distance_matrix[col, row] = distance_matrix[row, col]
        return distance_matrix


    ### Calculate residue distance (minimum distance between atoms for the given residues)
    def _calculate_residue_distance(self, residue_one, residue_two):
        distances = []
        for xyz_one in residue_one:
            for xyz_two in residue_two:
                diff2_xyz = (xyz_one - xyz_two)**2
                distance = np.sqrt(np.sum(diff2_xyz))
                distances.append(distance)
        return np.min(distances)


    ### Check for missing C-alphas in the chain
    def _check_Ca(self, plddt_mask, sequence):
        check = np.all(plddt_mask[:, 1])
        assert check, fr'Missing C-$\alpha$ for loaded sequence: {sequence}'


    ### Set modules
    def set_device(self, device):
        self.device = device
        torch.cuda.set_device(device)

    def set_distance_threshold(self, distance_threshold):
        self.distance_threshold


    ### Get modules
    def get_device(self): return self.device
    def get_distance_threshold(self): return self.distance_threshold







### -------------------------------------- MUTATION ALGORITHM ------------------------------------- ###
class Mutation_class(Basic_class):

    ### Initialization
    def __init__(
            self,
            wt_sequence : str,
            num_mutations : int = 10,
            sensitivity : float = 0.,
            T : float = 1.,
            gamma : float = 0.,
            results_dir : str = 'results',
            restart_bool : bool = False,
            device : int = 0,
            distance_threshold : float = 4.
    ):

        super().__init__(
                device = device,
                distance_threshold = distance_threshold
        )
        
        self.wt_sequence = wt_sequence
        self.wt_array = np.array(list(self.wt_sequence))    # to calculate distances
        self.wt_contacts = self.calculate_contacts(self.wt_sequence)
        
        self.distmatrix = pd.read_csv('DistPAM1.csv')
        self.distmatrix = self.distmatrix.drop(columns = ['Unnamed: 0'])
        self.residues = tuple(self.distmatrix.columns)
        self.distmatrix = np.array(self.distmatrix)

        if num_mutations > 0: self.num_mutations = num_mutations
        else: raise ValueError("Mutation_class.__init__(): num_mutations must be positive.")

        if sensitivity >= 0. and sensitivity <= 1.: self.sensitivity = sensitivity
        else: raise ValueError("Mutation_class.__init__(): sensitivity must be included in [0, 1].")

        if T > 0.: self.T = T
        else: raise ValueError("Mutation_class.__init__(): T must be positive.")

        if gamma >= 0.: self.gamma = gamma
        else: raise ValueError("Mutation_class.__init__(): gamma can't be negative.")

        self._get_id()
        self._check_directory(results_dir)
        self.set_restart_bool(restart_bool)
        self.print_status()



    ### Prepare filename simulation id
    def _get_id(self):
        T_str = str(self.T)[:1] + str(self.T)[2:7]
        s_str = str(self.sensitivity)[:1] + str(self.sensitivity)[2:7]
        g_str = str(self.gamma)[:1] + str(self.gamma)[2:7]
        self.file_id = 'T' + T_str + '_s' + s_str + '_g' + g_str



    ### Check for data directory to store data
    def _check_directory(self, results_dir):
        if results_dir[-1] != '/' and results_dir[:4] != '../':
            self.results_dir = results_dir
        else:
            if results_dir[-1] == '/':
                self.results_dir = results_dir[:-1]
            if results_dir[:4] == '../':
                self.results_dir = results_dir[4:]
        
        path = self.results_dir.split('/')
        actual_dir = '..'
        for idx, new_dir in enumerate(path):
            if idx > 0:
                actual_dir = actual_dir + '/' + path[idx - 1]
            print(new_dir, actual_dir)
            onlydirs = [d for d in listdir(f'{actual_dir}') if isdir(f'{actual_dir}/{d}')]
            if (new_dir in onlydirs) == False: 
                os.mkdir(f'{actual_dir}/{new_dir}')



    ### Reset parameters for new simulation
    def _reset(self, typ = 'total'):
        if typ == 'total':
            self.last_sequence = self.wt_sequence
            self.last_contacts = self.wt_contacts
            self.generation = 0
            self.accepted_mutations = 0
            self.last_eff_energy = 0
            self.last_ddG = 0
            self.last_PAM1_distance = 0
            self.last_Hamm_distance = 0
        elif typ == 'partial':
            self.generation = 0
            self.accepted_mutations = 0
        else:
            raise ValueError("Mutation_class._reset(): typ can only assume two values: 'total', 'partial'.")
        
        paths = [f'../{self.results_dir}/mutants_{self.file_id}.dat', f'../{self.results_dir}/data_{self.file_id}.dat', f'../{self.results_dir}/status_{self.file_id}.txt']
        onlyfiles = [f'../{self.results_dir}/{f}' for f in listdir(f'../{self.results_dir}') if isfile(f'../{self.results_dir}/{f}')]

        for path in paths:
            if path in onlyfiles:
                call(['rm', path])



    ### Restart the previous simulation
    def _restart(self, typ = 'total'):
        # Find files
        paths = [f'../{self.results_dir}/mutants_{self.file_id}.dat', f'../{self.results_dir}/data_{self.file_id}.dat']
        onlyfiles = [f'../{self.results_dir}/{f}' for f in listdir(f'../{self.results_dir}') if isfile(f'../{self.results_dir}/{f}')]
        check = np.all( [path in onlyfiles for path in paths] )

        if check:
            # Discard incomplete data
            with open(f'../{self.results_dir}/mutants_{self.file_id}.dat', 'r') as mutants_file:
                muts_lines = mutants_file.readlines()
                muts_generation = len(muts_lines)

            with open(f'../{self.results_dir}/data_{self.file_id}.dat', 'r') as data_file:
                data_lines = data_file.readlines()
                data_generation = len(data_lines)

            if muts_generation < data_generation:
                self.generation = muts_generation - 1   # do not count wt sequence
                with open(f'../{self.results_dir}/data_{self.file_id}.dat', 'w') as data_file:
                    for line in data_lines[:self.generation + 1]: 
                        print(line, end = '', file = data_file)

            elif muts_generation > data_generation:
                self.generation = data_generation - 1   # do not count wt sequence
                with open(f'../{self.results_dir}/mutants_{self.file_id}.dat', 'w') as mutants_file:
                    for line in muts_lines[:self.generation + 1]: 
                        print(line, end = '', file = mutants_file)

            elif muts_generation == data_generation:
                self.generation = muts_generation - 1


            # Last sequence
            with open(f'../{self.results_dir}/mutants_{self.file_id}.dat', 'r') as mutants_file:
                last_line = mutants_file.readlines()[-1].split('\t')
                self.last_sequence = last_line[1]
                if self.last_sequence[-1] == '\n': self.last_sequence = self.last_sequence[:-1]
                self.last_contacts = self.calculate_contacts(self.last_sequence)


            # Last sequence data
            with open(f'../{self.results_dir}/data_{self.file_id}.dat', 'r') as data_file:
                last_line = data_file.readlines()[-1].split('\t')
                self.last_eff_energy = float( last_line[1] )
                self.last_ddG = float( last_line[2] )
                self.last_PAM1_distance = float( last_line[3] )
                self.last_Hamm_distance = int( last_line[4] )
                self.accepted_mutations = int( float(last_line[5]) * self.generation )
                self.sensitivity = float( last_line[6] )
                self.T = float( last_line[7] )
                self.gamma = float( last_line[8] )

        else:
            self._reset(typ)



    ### Calculate Hamming distance and PAM1 distance
    def get_distances(self, new_sequence):
        new_array = np.array(list(new_sequence))
        mut_residues_idxs = np.where(self.wt_array != new_array)[0]

        # Hamming distance
        Hamm_distance = len(mut_residues_idxs)

        # PAM1 distance
        old_residues = self.wt_array[mut_residues_idxs]
        new_residues = new_array[mut_residues_idxs]
        PAM1_distance = 0.
        for old, new in zip(old_residues, new_residues):
            old_idx = self.residues.index(old)
            new_idx = self.residues.index(new)
            PAM1_distance += self.distmatrix[new_idx, old_idx]

        return PAM1_distance, Hamm_distance



    ### Produce single-residue mutation of the last metropolis sequence and calculate Hamming distance from wild-type protein sequence
    def single_mutation(self):
        # New sequence
        position = np.random.randint(0, len(self.wt_sequence))
        residue = self.residues[ np.random.randint(0, len(self.residues)) ]

        if residue == self.last_sequence[position]:
            # Repeat if no mutation occurred
            new_sequence = self.single_mutation()
            return new_sequence

        else:
            # Modify last sequence
            new_sequence = self.last_sequence[:position] + residue + self.last_sequence[(position + 1):]
            return new_sequence



    ### Calculate effective as number of modified contacts divided by the number of the wild-type protein contacts
    def calculate_effective_energy(self, mt_contacts):
        # Modified contacts fraction
        mod_diff = abs(mt_contacts - self.wt_contacts)
        norm = np.sum(mt_contacts) + np.sum(self.wt_contacts)
        eff_en = np.sum(mod_diff) / norm
        return eff_en



    ### Calculate ddG
    def calculate_ddG(self):
        pass



    ### Metropolis algorithm
    def metropolis(self):
        # Open files
        mutants_file = open(f'../{self.results_dir}/mutants_{self.file_id}.dat', 'a')
        data_file = open(f'../{self.results_dir}/data_{self.file_id}.dat', 'a')
        if self.generation == 0: 
            print(f'{self.generation}\t{self.last_sequence}', file = mutants_file)
            print(f'{self.generation}\t{format(self.last_eff_energy, ".15f")}\t{format(self.last_ddG, ".15f")}\t{self.last_PAM1_distance}\t{self.last_Hamm_distance}\t{float(self.accepted_mutations)}\t{self.sensitivity}\t{self.T}\t{self.gamma}\t{len(self.wt_sequence)}', file = data_file)


        # Metropolis
        for imut in range(self.num_mutations):
            # Mutant generation
            self.generation += 1
            new_sequence = self.single_mutation()
            mt_contacts = self.calculate_contacts(new_sequence)
    
            # Observables
            eff_energy = self.calculate_effective_energy(mt_contacts)
            ddG = 0
            PAM1_distance, Hamm_distance = self.get_distances(new_sequence)

            # Update lists
            p = np.random.rand()
            if p < np.exp( - (eff_energy - self.sensitivity) / self.T  - self.gamma * PAM1_distance ):
                self.last_sequence = new_sequence
                self.last_contacts = mt_contacts
                self.last_eff_energy = eff_energy
                self.last_ddG = ddG
                self.last_PAM1_distance = PAM1_distance
                self.last_Hamm_distance = Hamm_distance
                self.accepted_mutations += 1

            # Save data
            print(f'{self.generation}\t{self.last_sequence}', file = mutants_file)
            print(f'{self.generation}\t{format(self.last_eff_energy, ".15f")}\t{format(self.last_ddG, ".15f")}\t{self.last_PAM1_distance}\t{self.last_Hamm_distance}\t{self.accepted_mutations / self.generation}\t{self.sensitivity}\t{self.T}\t{self.gamma}\t{len(self.wt_sequence)}', file = data_file)

            # Print and save simulation status
            if self.generation % 100 == 0:
                self.print_last_mutation(f'../{self.results_dir}/status_{self.file_id}.txt')

        # Close data files
        mutants_file.close()
        data_file.close()



    ### Multi-metropolis algorithm, gamma variation
    def multimetropolis(self, gammas = []):
        if type(gammas) == float:
            gammas = [gammas]
        elif len(gammas) == 0:
            gammas = [self.gamma]

        # Simple metropolis
        if len(gammas) == 1:
            self.set_gamma(gammas[0])
            self.set_restart_bool(self.restart_bool, typ = 'total')
            print(f'gamma: {self.gamma} (1/1)')
            self.metropolis()

        # Actual multi-metropolis
        else:
            if self.restart_bool == True:
                print('Restart option is not available for multimetropolis. Setting "restart_bool = False".')

            self.restart_bool = False
            for idx, gamma in enumerate(gammas):
                print(f'gamma: {gamma} ({idx + 1}/{len(gammas)})')
                self.set_gamma(gamma)
                if idx == 0: typ = 'total'
                else: typ = 'partial'
                self.set_restart_bool(self.restart_bool, typ = typ)
                self.metropolis()



    ### Print status
    def print_status(self):
        print(f'Simulation PID: {os.getpid()}\n')
        
        print(f'Mutation algorithm protein:')
        print(f'Sequence: {self.wt_sequence}\n')

        print(f'Mutation algorithm parameters:')
        print(f'number of mutations: {self.num_mutations}')
        print(f'sensitivity:         {self.sensitivity}')
        print(f'temperature:         {self.T}')
        print(f'gamma:               {self.gamma}')
        print(f'distance threshold:  {self.distance_threshold} [A]')
        print(f'results directory:   ../{self.results_dir}\n')



    ### Print last mutation
    def print_last_mutation(self, print_file = sys.stdout):
        if print_file != sys.stdout:
            print_file = open(print_file, 'a')

        print(f'Generation:  {self.generation}', file = print_file)
        print(f'Wild tipe:   {self.wt_sequence}', file = print_file)
        print(f'Last mutant: {self.last_sequence}', file = print_file)
        print(f'Effective energy: {self.last_eff_energy}', file = print_file)
        print(f'ddG:              {self.last_ddG}', file = print_file)
        print(f'PAM1 distance:    {self.last_PAM1_distance}\n', file = print_file)
        print(f'Hamming distance: {self.last_Hamm_distance}\n', file = print_file)

        if print_file != sys.stdout:
            print_file.close()



    ### Set modules
    def set_wt_sequence(self, wt_sequence):
        self.wt_sequence = wt_sequence
        self.wt_contacts = self.calculate_contacts(self.wt_sequence)
        self._reset()

    def set_num_mutations(self, num_mutations): 
        self.num_mutations = num_mutations

    def set_T(self, T):
        if T > 0.: 
            self.T = T
            self._get_id()
        else: 
            raise ValueError("Mutation_class.__init__(): T must be positive.")

    def set_gamma(self, gamma):
        if gamma >= 0.: 
            self.gamma = gamma
            self._get_id()
        else: 
            raise ValueError("Mutation_class.__init__(): gamma can't be negative.")

    def set_restart_bool(self, restart_bool, typ = 'total'):
        self.restart_bool = restart_bool
        if self.restart_bool: self._restart(typ)
        else: self._reset(typ)



    ### Get modules
    def get_wt_sequence(self): return self.wt_sequence
    def get_wt_contacts(self): return self.wt_contacts
    def get_num_mutations(self): return self.num_mutations
    def get_sensitivity(self): return self.sensitivity
    def get_T(self): return self.T
    def get_gamma(self): return self.gamma
    def get_restart_bool(self): return self.restart_bool
    def get_generation(self): return self.generation
    def get_last_eff_energy(self): return self.last_eff_energy
    def get_last_ddG(self): return self.last_ddG
    def get_last_PAM1_distance(self): return self.last_PAM1_distance
    def get_last_Hamm_distance(self): return self.last_Hamm_distance
    def get_last_sequence(self): return self.last_sequence
    def get_last_contacts(self): return self.last_contacts
    def get_distmatrix(self): return self.distmatrix
    def get_residues(self): return self.residues
