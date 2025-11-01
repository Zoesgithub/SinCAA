# -*- coding: utf-8 -*-
"""
@File   :  chemical_feature.py
@Time   :  2024/02/05 19:34
@Author :  Yufan Liu
@Desc   :  Chemical features for network
"""

import numpy as np
from Bio.PDB import *
from scipy.spatial import KDTree

ele2num = {
    "C": 0,
    "H": 1,
    "O": 2,
    "N": 3,
    "S": 4,
    "SE": 5,
    "F": 6,
    "CL": 6,
    "BR": 6,
    "I": 6,
}


def generate_charge(pdb_filename, vertices):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_filename, pdb_filename)
    struct = struct[0]

    atom_types = []
    atom_coords = []
    for chain in struct:
        for residue in chain:
            for atom in residue:
                atom_types.append(atom.get_id())
                atom_coords.append(atom.get_coord())

    atom_coords = np.stack(atom_coords)
    assert atom_coords.shape[1] == 3

    kdt = KDTree(atom_coords)
    dists, results = kdt.query(vertices)
    atom_types = np.array(atom_types)[results]

    atom_feature = []
    for name in atom_types:

        if name[:2] in ele2num:
            atom_ = name[:2]
        else:
            atom_ = name[0]
        try:
            atom_feature.append(ele2num[atom_])
        except:
            atom_feature.append(7)

    return dists, np.array(atom_feature)
