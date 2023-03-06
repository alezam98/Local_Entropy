#!/usr/bin/env python3
import numpy as np
import Bio.PDB as PDB
import subprocess
import argparse
import sys
from warnings import warn


names = ['ALA', 'ASX', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'XAA', 'TYR', 'GLX']
    
def drop_het_residues(chain):
	"""Returns a chain without the het residues"""
	dropped_residue_ids = [residue.get_id() for residue in chain if residue.get_id()[0] != ' ']
	for dropped_residue_id in dropped_residue_ids:
		chain.detach_child(dropped_residue_id)
	return chain

def main(args):
    path = '/home/alessandroz/Desktop/contacts'
    pdb_code = args.pdb_code
    pdb_filename = f'{path}/files/{pdb_code}.pdb'
    
    sys.stderr = open(f'{path}/run/warnings/stderr.out', 'w')
    
    # Construction warnings check
    structure = PDB.PDBParser().get_structure(pdb_code, pdb_filename)
    model = structure[0]
    # Chains number check
    if len(model) > 1: warn("WARNING: ChainWarning: More than one chain identified in the pdb file.")
    # Residue names check
    chain = model.child_list[0]
    chain = drop_het_residues(chain)
    mask = [residue.resname in names for residue in chain]
    if False in mask: warn("WARNING: ResidueWarning: Unidentified residue name(s) in the analyzed chain.")
    
    sys.stderr.close()
    
    

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb-code",
        type = str,
        help = "PDB file containing structure information."
    )
    return parser
    
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
