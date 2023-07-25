#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.special import softmax
from jax.tree_util import tree_map
import os, glob, sys
from subprocess import call
from os import listdir
from os.path import isfile, isdir

import torch
import esm

from utils import randomize_sequence


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
        print(f'model:              esmfold_v1')
        print(f'device:             {self.device}')
        print(f'distance threshold: {self.distance_threshold} [A]', '\n')


    ### Calculate contact map through esmfold model
    def calculate_contacts(self, sequence, method = 'explicit', return_plddt = False, return_trivial = False):
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
        
        if not return_trivial: 
            contact_map = self._eliminate_trivial_contacts(contact_map)
        
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


    ### Eliminate trivial contacts from the contact map
    def _eliminate_trivial_contacts(self, contact_map):
        for row in range(len(contact_map)):
            contact_map[row, row] = 0
            if row > 0: contact_map[row, row - 1] = 0
            if row < len(contact_map) - 1: contact_map[row, row + 1] = 0
        return contact_map


    ### Set modules
    def set_device(self, device):
        self.device = device
        torch.cuda.set_device(device)

    def set_distance_threshold(self, distance_threshold):
        self.distance_threshold






### -------------------------------------- MUTATION ALGORITHM ------------------------------------- ###
class Mutation_class(Basic_class):

    ### Initialization
    def __init__(
            self,
            wt_sequence : str,
            mutations : int = 10,
            T : float = 1.,
            gamma : float = 1.,
            y : int = 1,
            seed : int = 0,
            unique_length : int = 10000,
            results_dir : str = 'results',
            step : int = 1,
            restart : bool = False,
            load : bool = False,
            comb : int = 0,
            device : int = 0,
            distance_threshold : float = 4.
    ):

        super().__init__(
                device = device,
                distance_threshold = distance_threshold
        )
        
        # Sequences
        self.wt_sequence = wt_sequence
        self.wt_contacts = self.calculate_contacts(self.wt_sequence)
        self.sequence_length = len(self.wt_sequence)

        # Distance definitions
        self.PAM1_distance_matrix = pd.read_csv('inputs/DistPAM1.csv')
        self.PAM1_distance_matrix = self.PAM1_distance_matrix.drop(columns = ['Unnamed: 0'])
        self.residues = tuple(self.PAM1_distance_matrix.columns)
        self.PAM1_distance_matrix = np.array(self.PAM1_distance_matrix)

        # Parameters
        if mutations > 0: self.mutations = mutations
        else: raise ValueError("Mutation_class.__init__(): mutations must be a positive integer.")

        if T > 0.: self.T = T
        else: raise ValueError("Mutation_class.__init__(): T must be positive.")

        if gamma >= 0.: self.gamma = gamma
        else: raise ValueError("Mutation_class.__init__(): gamma can't be negative.")

        if y > 0: self.y = y
        else: raise ValueError("Mutation_class.__init__(): y must be a positive integer.")

        self.seed = seed
        np.random.seed(self.seed)

        if unique_length >= self.y: self.unique_length = unique_length
        else: raise ValueError("Mutation_class.__init__(): unique_length can't be less than the number of replicas.")

        if step >= 1: self.step = step
        else: raise ValueError("Mutation_class.__init__(): step can't be less than 1.")

        # Initialization
        self.restart = restart
        self.load = load
        self.comb = comb
        
        self._check_directory(results_dir)
        if self.restart: self._restart()
        else: self._reset()



    ### Check for directory to store simulation mutants and data
    def _check_directory(self, results_dir):
        if results_dir[-1] != '/' and results_dir[:2] != './' and results_dir[:3] != '../':
            self.results_dir = results_dir
        else:
            if results_dir[-1] == '/':
                self.results_dir = results_dir[:-1]
            if results_dir[:2] == './':
                self.results_dir = results_dir[2:]
            if results_dir[:3] == '../':
                self.results_dir = results_dir[3:]
        
        path = self.results_dir.split('/')
        actual_dir = '..'
        for idx, new_dir in enumerate(path):
            if idx > 0:
                actual_dir = actual_dir + '/' + path[idx - 1]
            onlydirs = [d for d in listdir(f'{actual_dir}') if isdir(f'{actual_dir}/{d}')]
            if (new_dir in onlydirs) == False: 
                os.mkdir(f'{actual_dir}/{new_dir}')



    ### Define starting replicas
    def _starting_replicas(self):
        self.replica_sequences, self.replica_energies = np.array([]), np.array([])
        self.replica_distance_matrix = {
                'PAM1': np.zeros((self.y, self.y)),
                'Hamm': np.zeros((self.y, self.y))
        }

        if self.load:
            with open(f'inputs/combs/comb_{self.comb}.dat', 'r') as f:
                lines = f.readlines()
            assert len(lines[:-2]) == self.y, "Number of loaded replicas and y do not match."
            lengths = [len(line[:-1]) for line in lines[:-2]]
            assert len(np.unique(lengths)) == 1, "Loaded replicas with different lengths."
        else:
            random_sequence = randomize_sequence(self.residues, self.wt_sequence, fraction = 1.)

        for i in range(self.y):
            if self.load: replica = lines[i][:-1]
            else: replica = randomize_sequence(self.residues, random_sequence, fraction = 1.)
            contacts = self.calculate_contacts(replica)
            energy = self.calculate_energy(contacts)
            
            self.replica_sequences = np.append(self.replica_sequences, [replica])
            self.replica_energies = np.append(self.replica_energies, [energy])
            for j in range(i + 1):
                if j == i: continue
                PAM1_distance, Hamm_distance = self.calculate_distances(self.replica_sequences[i], self.replica_sequences[j])
                self.replica_distance_matrix['PAM1'][i, j] = PAM1_distance
                self.replica_distance_matrix['PAM1'][j, i] = PAM1_distance
                self.replica_distance_matrix['Hamm'][i, j] = Hamm_distance
                self.replica_distance_matrix['Hamm'][j, i] = Hamm_distance

        if self.load:
            eq_energy = float(lines[-1].split('\t')[-1])
            assert np.mean(self.replica_energies) <= eq_energy, "Loaded replicas mean energy does not satisfy equilibration constraint."

        # Save starting replicas
        path_README = '/'.join(self.results_dir.split('/')[:-1])
        check_README = np.any([f'../{path_README}/README.txt' == f'../{path_README}/{f}' for f in listdir(f'../{path_README}') if isfile(f'../{path_README}/{f}')])
        if not check_README:
            with open(f'../{path_README}/README.txt', 'w') as f:
                for replica_idx in range(self.y):
                    print(f'Replica {replica_idx}: {self.replica_sequences[replica_idx]}', file = f)



    ### Reset parameters for new simulation
    def _reset(self):
        self._starting_replicas()
        self.generation = 0
        self.accepted_mutations = 0
        
        self.unique_sequences = np.unique(self.replica_sequences)
        self.unique_energies = np.array([], dtype = float)
        for replica in self.unique_sequences:
            ireplica, = np.where(self.replica_sequences == replica)
            ireplica = ireplica[:1]
            self.unique_energies = np.append(self.unique_energies, self.replica_energies[ireplica])

        saved_files = [f'../{self.results_dir}/{f}' for f in listdir(f'../{self.results_dir}') if isfile(f'../{self.results_dir}/{f}')]
        for saved_file in saved_files:
            command = f'rm {saved_file}'.split(' ')
            call(command)



    ### Restart the previous simulation
    def _restart(self):
        # Find files
        saved_files = [f'../{self.results_dir}/{f}' for f in listdir(f'../{self.results_dir}') if isfile(f'../{self.results_dir}/{f}')]
        
        if len(saved_files) > 0:
            assert len(saved_files) == self.y + 1, 'Wrong number of saved files.'

            # Discard incomplete data
            lines = {}
            lengths = []
            for replica_idx in range(self.y):
                with open(f'../{self.results_dir}/replica_{replica_idx}.dat', 'r') as f:
                    lines[f'{replica_idx}'] = f.readlines()
                    lengths.append(len(lines[f'{replica_idx}']))

            min_length = np.min(lengths)
            for replica_idx in range(self.y):
                if lengths[replica_idx] > min_length:
                    with open(f'../{self.results_dir}/replica_{replica_idx}.dat', 'w') as f:
                        for line in lines[f'{replica_idx}'][:min_length]:
                            print(line, end = '', file = f)

            # Last replicas
            self.replica_sequences, self.replica_energies = np.array([]), np.array([])
            self.replica_distance_matrix = {
                    'PAM1': np.zeros((self.y, self.y)),
                    'Hamm': np.zeros((self.y, self.y))
            }
            for replica_idx in range(self.y):
                last_line = lines[f'{replica_idx}'][-1]
                splitted_line = last_line.split('\t')
                
                self.replica_sequences = np.append(self.replica_sequences, [splitted_line[1]])
                self.replica_energies = np.append(self.replica_energies, [float(splitted_line[2])])
                self.replica_distance_matrix['PAM1'][replica_idx, :] = [float(el) for el in splitted_line[3:(3+self.y)]]
                self.replica_distance_matrix['Hamm'][replica_idx, :] = [float(el) for el in splitted_line[(3+self.y):(3+2*self.y)]]

            # Unique replicas
            sequences, energies = np.array([]), np.array([])
            for replica_idx in range(self.y):
                sequences = np.append(sequences, [line.split('\t')[1] for line in lines[f'{replica_idx}']])
                energies = np.append(energies, [float(line.split('\t')[2]) for line in lines[f'{replica_idx}']])

            self.unique_sequences = np.unique(sequences)
            if len(self.unique_sequences) > self.unique_length:
                self.unique_sequences = self.unique_sequences[(len(self.unique_sequences) - self.unique_length):]

            idxs = [np.where(unique_sequence == sequences)[0][0] for unique_sequence in self.unique_sequences]
            self.unique_energies = np.array([energies[idx] for idx in idxs])
            assert len(self.unique_sequences) == len(self.unique_energies), 'Mismatch between the unique lists.'

            # Parameters
            with open(f'../{self.results_dir}/parameters.dat', 'r') as f:
                lines = f.readlines()
            last_line = lines[-1]
            splitted_line = last_line.split('\t')

            self.generation = int(splitted_line[0])
            self.T = float(splitted_line[1])
            self.gamma = float(splitted_line[2])
            self.y = int(splitted_line[3])
            self.seed = int(splitted_line[4])
            self.accepted_mutations = int( float(splitted_line[5]) * self.generation ) 
            assert self.sequence_length == int(splitted_line[6]), 'Mismatch between saved and actual sequence length.'

        else:
            self._reset()



    ### Calculate Hamming distance and PAM1 distance between sequences
    def calculate_distances(self, sequence_a, sequence_b):
        array_a = np.array(list(sequence_a))
        array_b = np.array(list(sequence_b))
        new_residues_idxs = np.where(array_a != array_b)[0]

        # Hamming distance
        Hamm_distance = len(new_residues_idxs) / self.sequence_length

        # PAM1 distance
        residues_a = array_a[new_residues_idxs]
        residues_b = array_b[new_residues_idxs]
        PAM1_distance = 0.
        for residue_a, residue_b in zip(residues_a, residues_b):
            idx_a = self.residues.index(residue_a)
            idx_b = self.residues.index(residue_b)
            PAM1_distance += self.PAM1_distance_matrix[idx_a, idx_b]
        PAM1_distance = PAM1_distance / self.sequence_length
        
        return PAM1_distance, Hamm_distance



    ### Produce single-residue mutation of the last metropolis sequence
    def single_mutation(self, sequence):
        # New residue
        position = np.random.randint(0, self.sequence_length)
        residue = self.residues[ np.random.randint(0, len(self.residues)) ]

        if residue == sequence[position]:
            # Repeat if no mutation occurred
            mt_sequence = self.single_mutation(sequence)
            return mt_sequence

        else:
            # Generate mutant from last sequence
            mt_sequence = sequence[:position] + residue + sequence[(position + 1):]
            return mt_sequence



    ### Calculate effective as number of modified contacts divided by the number of the wild-type protein contacts
    def calculate_energy(self, mt_contacts):
        # Modified contacts fraction
        mod_diff = abs(mt_contacts - self.wt_contacts)
        norm = np.sum(mt_contacts) + np.sum(self.wt_contacts)
        energy = np.sum(mod_diff) / norm
        return energy



    ### Calculate ddG
    def calculate_ddG(self):
        pass



    ### Metropolis algorithm
    def metropolis(self, save = True, print_progress = True):
        # Starting status print
        if save and self.generation == 0: self._save()
        if print_progress: self.print_progress()

        # Metropolis
        for imut in range(self.mutations):
            # Mutant generation
            self.generation += 1
            replica_idx = np.random.randint(0, self.y)
            mt_sequence = self.single_mutation(self.replica_sequences[replica_idx])

            # Observables
            mask = self.unique_sequences == mt_sequence
            assert np.sum(mask.astype(int)) <= 1, "Too many 'unique' sequences equal to the same mutant."
            if np.any(mask):
                assert self.unique_sequences[mask][0] == mt_sequence, 'Wrong mask.'
                mt_energy = self.unique_energies[mask][0]
            else:
                mt_contacts = self.calculate_contacts(mt_sequence)
                mt_energy = self.calculate_energy(mt_contacts)

                self.unique_sequences = np.append(self.unique_sequences, mt_sequence)
                self.unique_energies = np.append(self.unique_energies, [mt_energy])
                assert len(self.unique_sequences) == len(self.unique_energies), "Length of unique sequences and unique energies must coincide."
                if len(self.unique_sequences) > self.unique_length:
                    self.unique_sequences = self.unique_sequences[1:]
                    self.unique_energies = self.unique_energies[1:]
            
            new_distance_matrix = {}
            for key in self.replica_distance_matrix: new_distance_matrix[key] = self.replica_distance_matrix[key].copy()
            for i in range(self.y):
                if i == replica_idx: continue
                PAM1_distance, Hamm_distance = self.calculate_distances(self.replica_sequences[i], mt_sequence)
                new_distance_matrix['PAM1'][i, replica_idx] = PAM1_distance
                new_distance_matrix['PAM1'][replica_idx, i] = PAM1_distance
                new_distance_matrix['Hamm'][i, replica_idx] = Hamm_distance
                new_distance_matrix['Hamm'][replica_idx, i] = Hamm_distance
            
            # Update lists
            dE = mt_energy - self.replica_energies[replica_idx]
            dd = (new_distance_matrix['PAM1'][replica_idx, :] - self.replica_distance_matrix['PAM1'][replica_idx, :]).sum()
            p = np.random.rand()
            if p <= np.exp( -dE/self.T -self.gamma*dd ):
                self.replica_sequences[replica_idx] = mt_sequence
                self.replica_energies[replica_idx] = mt_energy
                for key in self.replica_distance_matrix: self.replica_distance_matrix[key] = new_distance_matrix[key].copy()
                self.accepted_mutations += 1

            # Save data
            if save and (self.generation%self.step == 0): self._save()
            if print_progress and (self.generation%1000 == 0): self.print_progress()



    ### Print status
    def print_status(self):
        print(f'Mutation algorithm protein:')
        print(f'Wild-type sequence: {self.wt_sequence}')
        print(f'Starting replicas:')
        for ireplica, replica in enumerate(self.replica_sequences): 
            print(f'replica {ireplica}: {replica}')
        print()

        print(f'Mutation algorithm parameters:')
        print(f'mutations:           {self.mutations}')
        print(f'current generation:  {self.generation}')
        print(f'temperature:         {self.T}')
        print(f'gamma:               {self.gamma}')
        print(f'y:                   {self.y}')
        print(f'seed:                {self.seed}')
        print(f'unique length:       {self.unique_length}')
        print(f'results directory:   ../{self.results_dir}')
        print(f'step:                {self.step}')
        print(f'restart:             {self.restart}\n')



    ### Print simulation progress
    def print_progress(self):
        print(f'- Generation: {self.generation}')
        print(f'- Replicas:')
        for ireplica, (replica, energy, distances) in enumerate(zip(self.replica_sequences, self.replica_energies, self.replica_distance_matrix['Hamm'])): 
            print(f'{ireplica}\t{replica}\t{energy}\t{distances.sum()/(self.y - 1)}')
        print()



    ### Save data
    def _save(self):
        # Save replicas
        for replica_idx in range(self.y):
            with open(f'../{self.results_dir}/replica_{replica_idx}.dat', 'a') as f:
                line = f'{self.generation}\t'
                line = line + f'{self.replica_sequences[replica_idx]}\t'
                line = line + f'{self.replica_energies[replica_idx]}\t'
                line = line + '\t'.join(self.replica_distance_matrix['PAM1'][replica_idx, :].astype(str)) + '\t'
                line = line + '\t'.join(self.replica_distance_matrix['Hamm'][replica_idx, :].astype(str))
                print(line, file = f)

        # Save parameters
        with open(f'../{self.results_dir}/parameters.dat', 'a') as f:
            line = f'{self.generation}\t'
            line = line + f'{self.T}\t'
            line = line + f'{self.gamma}\t'
            line = line + f'{self.y}\t'
            line = line + f'{self.seed}\t'
            if self.generation == 0: line = line + f'{self.accepted_mutations}\t'
            else: line = line + f'{self.accepted_mutations/self.generation}\t'
            line = line + f'{self.sequence_length}'
            print(line, file = f)



    ### Set modules
    def set_wt_sequence(self, wt_sequence : str):
        self.wt_sequence = wt_sequence
        self.wt_contacts = self.calculate_contacts(self.wt_sequence)
        self.sequence_length = len(self.wt_sequence)
        self._reset()

    def set_mutations(self, mutations : int): 
        if mutations > 0: self.mutations = mutations
        else: raise ValueError("Mutation_class.set_mutations(): mutations must be a positive integer.")

    def set_T(self, T : float):
        if T > 0.: self.T = T
        else: raise ValueError("Mutation_class.set_T(): T must be positive.")

    def set_gamma(self, gamma : float):
        if gamma >= 0.: self.gamma = gamma
        else: raise ValueError("Mutation_class.set_gamma(): gamma can't be negative.")

    def set_seed(self, seed : int):
        self.seed = seed
        np.random.seed(self.seed)

    def set_unique_length(self, unique_length : int):
        if unique_length >= 0: self.unique_length = unique_length
        else: raise ValueError("Mutation_class.set_unique_length(): unique_length can't be negative.")
        self.restart_bool()

    def set_results_dir(self, results_dir : str):
        self.results_dir = results_dir

    def set_step(self, step : int):
        if step >= 1: self.step = step
        else: raise ValueError("Mutation_class.set_step(): step can't be less than 1.")

    def set_restart(self, restart : bool):
        self.restart = restart
        if self.restart: self._restart()
        else: self._reset()
