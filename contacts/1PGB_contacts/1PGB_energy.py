import numpy as np
import matplotlib.pyplot as plt
import Bio.PDB as PDB
import argparse

import torch
import esm

names = ['ALA', 'ASX', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'XAA', 'TYR', 'GLX']
symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']



def check_model(model, args):
    """Check for number of chains, chain length and residue names"""
    model_check = len(model) == 1

    chain = model.child_list[0]
    chain_name = list(model.child_dict.keys())[model.child_list.index(chain)]
    modified_chain = drop_het_residues(chain)

    residue_names = [residue.resname for residue in modified_chain]
    names_check = np.all([residue.resname in names for residue in modified_chain])

    if model_check and names_check:        # do not skip
        return modified_chain, chain_name, residue_names
    else:                                                   # skip
        return None

def drop_het_residues(chain):
    """Returns a chain without the het residues"""
    dropped_residue_ids = [residue.get_id() for residue in chain if residue.get_id()[0] != ' ']
    for dropped_residue_id in dropped_residue_ids:
        chain.detach_child(dropped_residue_id)
    return chain
	
def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    coords_one = [residue_one[name].coord for name in residue_one.child_dict if not list(name)[0] == 'H']
    coords_two = [residue_two[name].coord for name in residue_two.child_dict if not list(name)[0] == 'H']
    
    distances = []
    for coord_one in coords_one:
        for coord_two in coords_two:
            diff_vector  = coord_one - coord_two
            distance = np.sqrt(np.sum(diff_vector * diff_vector))
            distances.append(distance)
    return np.min(distances)
	
def calc_dist_matrix(chain) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain), len(chain)), float)
    for row, residue_one in enumerate(chain) :
        for col, residue_two in enumerate(chain) :
            if col > row:
                answer[row, col] = calc_residue_dist(residue_one, residue_two)
                answer[col, row] = answer[row, col]
    return answer
    
def calc_eff_energy(PDB_contact_map, esm_contact_map):
    mod_diff = abs(PDB_contact_map - esm_contact_map)
    norm = np.sum(PDB_contact_map) + np.sum(esm_contact_map)
    eff_energy = np.sum(mod_diff) / norm
    return eff_energy



def main(args):
    # Load model
    print('Loading esm model...')
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()  # disables dropout for deterministic results

    pdb_filename = '1pgb.pdb'
    structure = PDB.PDBParser().get_structure('1pgb', '1pgb.pdb')
    model = structure[0]

    # Check and print protein properties
    output = check_model(model, args)
    if output == None:
        raise ValueError('Problem with model.')
    else:
        chain, chain_name, residue_names = output

    idxs = [names.index(name) for name in residue_names]
    wt_sequence_list = [symbols[idx] for idx in idxs]
    wt_sequence = ''.join(wt_sequence_list)
    
    # Calculate PDB contact map
    print('Calculating PDB contact map...')
    dist_matrix = calc_dist_matrix(chain)
    PDB_contact_map = (dist_matrix < args.distance_threshold).astype(int)

    print('Calculating esm predicted contact map, explicit method...')
    data = [
        ("1PGB", wt_sequence),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
    esm_contact_map = np.array(results["contacts"][: batch_lens[0], : batch_lens[0]][0]) > 0.5
    eff_energy = calc_eff_energy(PDB_contact_map, esm_contact_map)
    
    print(f'Protein 1PGB...')
    print(f'- number of contacts: {np.sum(PDB_contact_map)}')
    print(f'- mutated contacts:   {np.sum(abs(PDB_contact_map - esm_contact_map))}')
    print(f'- preserved contacts: {np.sum(PDB_contact_map) - np.sum(abs(PDB_contact_map - esm_contact_map))}')
    print(f'- effective energy: {eff_energy}')
    print()

    fig_esm = plt.figure()
    plt.matshow(esm_contact_map)
    plt.savefig('1PGB_esm.png', bbox_inches='tight')

    fig_pdb = plt.figure()
    plt.matshow(PDB_contact_map)
    plt.savefig('1PGB_pdb.png', bbox_inches='tight')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type = int,
        default = 0,
        help = "int variable, cuda device used to load the esmfold model. Possible choices: 0, 1, 2. Default: 0."
    )
    parser.add_argument(
    	"--distance-threshold",
    	type = float,
    	default = 4.,
    	help = "float variable, maximum distance which defines a contact. Default: 4.0 [A]."
    )
    return parser	


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
