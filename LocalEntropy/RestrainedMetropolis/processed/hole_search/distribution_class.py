import numpy as np
import pandas as pd
from time import time
import os
from os import listdir
from os.path import isfile, isdir



class Distribution_class:
    ### Initialization
    def __init__(
            self,
            T : float = 0.007,
            energy_threshold : float = 0.2,
            dtype : str = 'Hamm',
            step : int = 1,
            prec : float = 1.,
            discarded_mutations : int = 0,
            inputs_d : str = 'inputs',
            groups_d : str = 'groups',
            pdfs_d : str = 'pdfs',
            restart : bool = True,
    ):

        if T < 0.: raise ValueError("Incompatible value for T variable. Allowed values: T >= 0.")
        else: self.T = T

        if energy_threshold < 0.: raise ValueError("Incompatible value for energy_threshold. Allowed values: energy_threshold >= 0.")
        else: self.energy_threshold = energy_threshold

        self.inputs_d = inputs_d
        self.groups_d = groups_d
        self.pdfs_d = pdfs_d

        if not dtype in ("Hamm", "PAM1"): raise ValueError("Incompatible value for dtype variable. Allowed values: 'Hamm' (Hamming distance pdf), 'PAM1' (PAM1 distance pdf).")
        else: 
            self.dtype = dtype
            if self.dtype == "PAM1":
                distmatrix = pd.read_csv(f'{self.inputs_d}/DistPAM1.csv')
                distmatrix = distmatrix.drop(columns = ['Unnamed: 0'])
                self.residues = tuple(distmatrix.columns)
                self.distmatrix = np.array(distmatrix)

        if step < 1: raise ValueError("Incompatible value for step variable. Allowed values: step > 0.")
        else: self.step = step

        if prec < 0.: raise ValueError("Incompatible value for prec variable. Allowed values: prec > 0.")
        else:
            if self.dtype == 'Hamm': self.prec = 1.
            elif self.dtype == 'PAM1': self.prec = prec

        if discarded_mutations < 0: raise ValueError("Incompatible value for discarded_mutations variable. Allowed values: discarded_mutations >= 0.")
        else: self.discarded_mutations = discarded_mutations

        self._get_ids()

        self.restart = restart
        self.load_csv()
        self.load_filelist()



    ### Get groups and pdfs ids
    def _get_ids(self):
        self.groups_id = f'T{self.T}_e{self.energy_threshold}_dm{self.discarded_mutations}'
        self.pdfs_id = f'{self.dtype}_T{self.T}_e{self.energy_threshold}_p{self.prec}_s{self.step}_dm{self.discarded_mutations}'



    ### Load csv files for same and mutual pdfs
    def load_csv(self):
        if self.restart:
            same, mutual = False, False
            for f in listdir(self.pdfs_d):
                if isfile(f'{self.pdfs_d}/{f}') and (self.pdfs_id in f):
                    if 'same' in f:
                        self.same_pdfs = pd.read_csv(f'{self.pdfs_d}/{f}')
                        self.same_pdfs = self.same_pdfs.drop(columns = ['Unnamed: 0'])
                        same = True
                    elif 'mutual' in f:
                        self.mutual_pdfs = pd.read_csv(f'{self.pdfs_d}/{f}')
                        self.mutual_pdfs = self.mutual_pdfs.drop(columns = ['Unnamed: 0'])
                        mutual = True
            if not same:
                self.same_pdfs = pd.DataFrame()
            if not mutual:
                self.mutual_pdfs = pd.DataFrame()
            
            self.same_skip = np.array(self.same_pdfs.columns)
            self.mutual_skip = np.array(self.mutual_pdfs.columns)
        
        else:
            self.same_pdfs = pd.DataFrame()
            self.mutual_pdfs = pd.DataFrame()

            self.same_skip = []
            self.mutual_skip = []




    ### Load filelist from which load the groups
    def load_filelist(self):
        self.filelist = np.array([f for f in listdir(self.groups_d) if isfile(f'{self.groups_d}/{f}') and (self.groups_id in f)], dtype = str)



    ### Load group from filelist
    def load_group(self, filename : str):
        assert filename in self.filelist, "File not found."

        with open(f'{self.groups_d}/{filename}', 'r') as file:
            lines = file.readlines()
        group = np.array([line[:-1] for line in lines], dtype = str)
        
        lengths = np.unique([len(mutant) for mutant in group])
        assert len(lengths) == 1, f"Different protein lengths in the same file {filename}."
        return group[::self.step]



    ### Calculate PAM1 distance between sequences
    def calculate_PAM1_distance(self, mut1, mut2):
        diff_residues_idxs = np.where(mut1_array != mut2_array)[0]

        diff_residues1 = mut1_array[diff_residues_idxs]
        diff_residues2 = mut2_array[diff_residues_idxs]
        PAM1_distance = 0.
        for residue1, residue2 in zip(diff_residues1, diff_residues2):
            idx1 = self.residues.index(residue1)
            idx2 = self.residues.index(residue2)
            PAM1_distance += self.distmatrix[idx1, idx2]

        return PAM1_distance



    ### Calculate pdf with PAM1 distance
    def calculate_PAM1_distribution(self):
        assert len(self.group_a[0]) % self.prec == 0, 'Choose a precision value such that the protein length is divisible by it.'

        length = len(self.group_a[0])
        pdf = np.zeros(int(length / self.prec))
        t0, partial_t0 = time(), time()
    
        if len(self.group_b) == 0:
        
            for imut_a, mut_a in enumerate(self.group_a):
                if imut_a % 1000 == 0:                    
                    print(f'progress: {imut_a}/{len(self.group_a)}\ttime: {format(time() - t0, ".1f")} s\tpartial time: {format(time() - partial_t0, ".1f")}')
                    partial_t0 = time()
                
                mut_a = np.array(list(mut_a))
                mut_a_pdf = np.zeros(int(length / self.prec))
                for mut_b in self.group_a[(imut_a + 1):]:
                    mut_b = np.array(list(mut_b))
                    distance_ab = self.calculate_distance(mut_a, mut_b)
                    idistance_ab = int( (distance_ab - distance_ab % self.prec) / self.prec )
                    if idistance_ab == int(length / self.prec): idistance_ab = idistance_ab - 1
                    mut_a_pdf[idistance_ab] += 1

            pdf = pdf + mut_a_pdf
    
        else:
            assert len(self.group_a[0]) == len(self.group_b[0]), "Can't calculate distance between sequences with different length."
        
            for imut_a, mut_a in enumerate(self.group_a):
                if imut_a % 1000 == 0: 
                    print(f'progress: {imut_a}/{len(self.group_a)}\ttime: {format(time() - t0, ".1f")} s\tpartial time: {format(time() - partial_t0, ".1f")}')
                    partial_t0 = time()

                mut_a = np.array(list(mut_a))
                mut_a_pdf = np.zeros(int(length / self.prec))
                for mut_b in self.group_b:
                    mut_b = np.array(list(mut_b))
                    distance_ab = self.calculate_distance(mut_a, mut_b)
                    idistance_ab = int( (distance_ab - distance_ab % self.prec) / self.prec )
                    if idistance_ab == int(length / self.prec): idistance_ab = idistance_ab - 1
                    mut_a_pdf[idistance_ab] += 1

                pdf = pdf + mut_a_pdf

        print(f'progress: {len(self.group_a)}/{len(self.group_a)}\ttime: {format(time() - t0, ".1f")} s\tpartial time: {format(time() - partial_t0, ".1f")}')
        print(f'Total time: {format(time() - t0, ".1f")}')
        pdf = pdf / np.sum(pdf)
        return pdf



    ### Calculate pdf with Hamming distance
    def calculate_Hamm_distribution(self):
        length = len(self.group_a[0])
        pdf = np.zeros(length + 1)
        t0, partial_t0 = time(), time()

        if len(self.group_b) == 0:
        
            for imut_a, mut_a in enumerate(self.group_a):
                if imut_a % 1000 == 0:
                    print(f'progress: {imut_a}/{len(self.group_a)}\ttime: {format(time() - t0, ".1f")} s\tpartial time: {format(time() - partial_t0, ".1f")}')
                    partial_t0 = time()

                mut_a = np.array(list(mut_a))
                mut_a_pdf = np.zeros(length + 1)
                for mut_b in self.group_a[(imut_a + 1):]:
                    mut_b = np.array(list(mut_b))
                    distance_ab = len(np.where(mut_a != mut_b)[0])
                    mut_a_pdf[distance_ab] += 1

                pdf = pdf + mut_a_pdf

        else:
            assert len(self.group_a[0]) == len(self.group_b[0]), "Can't calculate distance between sequences with different length."

            for imut_a, mut_a in enumerate(self.group_a):
                if imut_a % 1000 == 0:
                    print(f'progress: {imut_a}/{len(self.group_a)}\ttime: {format(time() - t0, ".1f")} s\tpartial time: {format(time() - partial_t0, ".1f")}')
                    partial_t0 = time()

                mut_a = np.array(list(mut_a))
                mut_a_pdf = np.zeros(length + 1)
                for mut_b in self.group_b:
                    mut_b = np.array(list(mut_b))
                    distance_ab = len(np.where(mut_a != mut_b)[0])
                    mut_a_pdf[distance_ab] += 1

                pdf = pdf + mut_a_pdf

        print(f'progress: {len(self.group_a)}/{len(self.group_a)}\ttime: {format(time() - t0, ".1f")} s\tpartial time: {format(time() - partial_t0, ".1f")}')
        print(f'Total time: {format(time() - t0, ".1f")}')
        pdf = pdf / np.sum(pdf)
        return pdf



    ### Calculate pdfs from groups
    def get_group_distribution(self, pdftype : str):
        assert pdftype in ('same', 'mutual'), "Incompatible value for pdf_type variable. Allowed values: 'same', 'all'."
        
        if pdftype == 'same':
            self.group_b = []

            for ifile, filename in enumerate(self.filelist):
                group_num = filename.split('_')[1]
                column_name = f'group {group_num}'
                
                if (not self.restart) or (not column_name in self.same_skip):
                    print(f'- {column_name}:')
                    self.group_a = self.load_group(filename)
                    
                    if self.dtype == 'Hamm': pdf = self.calculate_Hamm_distribution()
                    elif self.dtype == 'PAM1': pdf = self.calculate_PAM1_distribution()

                    if (not 'distances' in self.same_pdfs.columns):
                        self.same_pdfs['distances'] = np.arange(len(pdf)) * self.prec

                    self.same_pdfs[column_name] = pdf
                    self.same_pdfs.to_csv(f'{self.pdfs_d}/pdf_{pdftype}_{self.pdfs_id}.csv')

        elif pdftype == 'mutual':
            if len(self.filelist) == 1:
                print('Not enough groups to calculate the mutual distances distribution.')

            for ifile_a, filename_a in enumerate(self.filelist):
                for ifile_b, filename_b in enumerate(self.filelist[(ifile_a + 1):]):
                    group_a_num = filename_a.split('_')[1]
                    group_b_num = filename_b.split('_')[1]
                    column_name = f'groups {group_a_num} - {group_b_num}'

                    if (not self.restart) or (not column_name in self.mutual_skip):
                        print(f'- {column_name}:')
                        if ifile_b == 0: self.group_a = self.load_group(filename_a)
                        self.group_b = self.load_group(filename_b)
                        
                        if self.dtype == 'Hamm': pdf = self.calculate_Hamm_distribution()
                        elif self.dtype == 'PAM1': pdf = self.calculate_PAM1_distribution()

                        if (not 'distances' in self.mutual_pdfs.columns):
                            self.mutual_pdfs['distances'] = np.arange(len(pdf)) * self.prec

                        self.mutual_pdfs[column_name] = pdf
                        self.mutual_pdfs.to_csv(f'{self.pdfs_d}/pdf_{pdftype}_{self.pdfs_id}.csv')



    ### Set modules
    def set_T(self, T : float):
        if T < 0.: raise ValueError("Incompatible value for T variable. Allowed values: T >= 0.")
        else: self.T = T
        self._get_ids()
        self.load_csv()
        self.load_filelist()

    def set_energy_threshold(self, energy_threshold : float):
        if energy_threshold < 0.: raise ValueError("Incompatible value for energy_threshold. Allowed values: energy_trheshold >= 0.")
        else: self.energy_threshold = energy_threshold
        self._get_ids()
        self.load_csv()
        self.load_filelist()

    def set_dtype(self, dtype : str):
        if not dtype in ("Hamm", "PAM1"): raise ValueError("Incompatible value for dtype variable. Allowed values: 'Hamm' (Hamming distance pdf), 'PAM1' (PAM1 distance pdf).")
        else:
            self.dtype = dtype
            if self.dtype == "PAM1":
                distmatrix = pd.read_csv(f'{self.inputs_d}/DistPAM1.csv')
                distmatrix = distmatrix.drop(columns = ['Unnamed: 0'])
                self.residues = tuple(distmatrix.columns)
                self.distmatrix = np.array(distmatrix)
        self._get_ids()
        self.load_csv()

    def set_step(self, step : int):
        if step < 1: raise ValueError("Incompatible value for step variable. Allowed values: step > 0.")
        else: self.step = step
        self._get_ids()
        self.load_csv()

    def set_prec(self, prec : float):
        if prec < 0.: raise ValueError("Incompatible value for prec variable. Allowed values: prec > 0.")
        else:
            if self.dtype == 'Hamm': self.prec = 1.
            elif self.dtype == 'PAM1': self.prec = prec
        self._get_ids()
        self.load_csv()

    def set_discarded_mutations(self, discarded_mutations : int):
        if discarded_mutations < 0: raise ValueError("Incompatible value for discarded_mutations variable. Allowed values: discarded_mutations >= 0.")
        else: self.discarded_mutations = discarded_mutations
        self._get_ids()
        self.load_csv()
        self.load_filelist()

    def set_inputs_d(self, inputs_d):
        self.inputs_d = inputs_d
        distmatrix = pd.read_csv(f'{self.inputs_d}/DistPAM1.csv')
        distmatrix = distmatrix.drop(columns = ['Unnamed: 0'])
        self.residues = tuple(distmatrix.columns)
        self.distmatrix = np.array(distmatrix)

    def set_groups_d(self, groups_d):
        self.groups_d = groups_d
        self.load_filelist()
        
    def set_pdfs_d(self, pdfs_d):
        self.pdfs_d = pdfs_d
        self.load_csv()

    def set_restart(self, restart : bool):
        self.restart = restart
        self.load_csv()



    ### Get modules
    def get_T(self): return self.T
    def get_energy_threshold(self): return self.energy_threshold
    def get_dtype(self): return self.dtype
    def get_step(self): return self.step
    def get_prec(self): return self.prec
    def get_discarded_mutations(self): return self.discarded_mutations
    def get_inputs_d(self): return self.inputs_d
    def get_groups_d(self): return self.groups_d
    def get_pdfs_d(self): return self.pdfs_d
    def get_restart(self): return self.restart
    def get_distmatrix(self): return self.distmatrix
    def get_residues(self): return self.residues
    def get_filelist(self): return self.filelist
    def get_same_pdfs(self): return self.same_pdfs
    def get_mutual_pdfs(self): return self.mutual_pdfs
    def get_groups_id(self): return self.groups_id
    def get_pdfs_id(self): return self.pdfs_id
