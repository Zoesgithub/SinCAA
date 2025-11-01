# -*- coding: utf-8 -*-
"""
@File   :  smiles_to_graph.py
@Time   :  2024/05/27 12:12
@Author :  Yufan Liu
@Desc   :  Mainly from torch geometric from_smiles
"""

from typing import List

import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.utils.smiles import e_map, x_map

from peptide.graph import ATOM_VOCAB


def from_smiles(sdf_path: str, with_hydrogen: bool = False, kekulize: bool = False):
    """Generate PyG graph from SMILES string, mainly from PyG

    Args:
        sdf_path (str): path for SDF file (from PDB)
        with_hydrogen (bool, optional)
        kekulize (bool, optional)

    Returns:
        PyG Data type
    """
    RDLogger.DisableLog("rdApp.*")
    sdf_supplier = Chem.SDMolSupplier(sdf_path)
    mol = sdf_supplier[0]
    smiles = Chem.MolToSmiles(mol)

    if mol is None:
        mol = Chem.MolFromSmiles("")
        raise NotImplementedError("SMILES is Empty")
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():
        row: List[int] = []
        atom_type = atom.GetSymbol()
        if not atom_type in ATOM_VOCAB:
            atom_type = "UNK"
        row.append(ATOM_VOCAB[atom.GetSymbol()])
        row.append(x_map["chirality"].index(str(atom.GetChiralTag())))
        row.append(x_map["degree"].index(atom.GetTotalDegree()))
        row.append(x_map["formal_charge"].index(atom.GetFormalCharge()))
        row.append(x_map["num_hs"].index(atom.GetTotalNumHs()))
        row.append(x_map["num_radical_electrons"].index(atom.GetNumRadicalElectrons()))
        row.append(x_map["hybridization"].index(str(atom.GetHybridization())))
        row.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
        row.append(x_map["is_in_ring"].index(atom.IsInRing()))
        xs.append(row)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map["bond_type"].index(str(bond.GetBondType())))
        e.append(e_map["stereo"].index(str(bond.GetStereo())))
        e.append(e_map["is_conjugated"].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles), mol
