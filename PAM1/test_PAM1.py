#!/usr/bin/env python
import numpy as np
import pandas as pd

with open('PAM1.txt', 'r') as file:
    lines = file.readlines()
    splitted_lines = [line.split(' ') for line in lines]

matrix = []
for idx, line in enumerate(splitted_lines):
    line = [elem for elem in line if elem != '']
    line[-1] = line[-1][:-1]
    if idx != 0:
        line.pop(0)
    matrix.append(line)

residues = matrix.pop(0)
print('residues:\n', residues, '\n')

matrix = np.array(matrix, dtype = float)
print('matrix:\n', matrix, '\n')

for idx in range(len(matrix)):
    colsum = matrix[:, idx].sum() - matrix[idx, idx]
    matrix[:, idx] = matrix[:, idx] / colsum
    matrix[idx, idx] = 1
print('new matrix:\n', matrix, '\n')

df_matrix = pd.DataFrame(matrix, columns = residues, index = residues)
print('dataframe:\n', df_matrix, '\n')
df_matrix.to_csv('ModPAM1.csv')

distmatrix = 1. - (matrix + np.transpose(matrix))/2.
print('distance matrix:\n', distmatrix, '\n')

df_distmatrix = pd.DataFrame(distmatrix, columns = residues, index = residues)
print('distance dataframe:\n', df_distmatrix, '\n')
df_distmatrix.to_csv('DistPAM1.csv')
