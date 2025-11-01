# -*- coding: utf-8 -*-
"""
@File   :  prepare_data.py
@Time   :  2024/05/27 16:29
@Author :  Yufan Liu
@Desc   :  Prepare data and store
"""

import dataclasses
import os
import os.path as osp
from typing import List

import Bio.PDB as biopdb
import numpy as np
import torch
from torch_geometric.data import Data

from peptide.graph import PeptideChain, mmcif_to_graph
from protein.surface import mmcif_to_surface
from utils.gt import generate_gt
from utils.smiles_to_graph import from_smiles


@dataclasses.dataclass()
class PeptideGraph:
    data_name: str
    peptide_edge_index: torch.Tensor
    residue_graph: List[Data]


def construct_graph(peptide_info: PeptideChain, chain):
    """Construct residue graphs

    The residue graphs contains two types of edges

    Args:
        peptide_info (PeptideChain): including subgraphs of residue
        chain (_type_): indicate which chain will be used

    Returns:
        A list of residue graphs in PyG Data class
    """
    graphs = []
    struct = peptide_info.mmcif_structure
    valid_res = peptide_info.mmcif_seqres_struct[chain]

    for ch in struct:
        if ch.id == chain:
            struct_res_id = [res.get_id() for res in ch]
            struct_res = [res for res in ch]

    for i in sorted(valid_res.keys()):
        res_flag = valid_res[i]
        if res_flag.position:  # residues without missing
            residue = struct_res[
                struct_res_id.index(
                    (
                        res_flag.hetflag,
                        res_flag.position.residue_number,
                        res_flag.position.insertion_code,
                    )
                )
            ]
            res_name = residue.resname
            res_graph, res_mol = from_smiles(
                f"raw_data/residues_lib/{res_name}_ideal.sdf"
            )

            # match sdf mol and cif mol, as order, thie following part is for generate a fully-connected graph
            res_cif = biopdb.MMCIF2Dict.MMCIF2Dict(
                f"raw_data/residues_lib/{res_name}.cif"
            )
            cif_atom_names = [
                res_cif["_chem_comp_atom.atom_id"][i]
                for i in range(len(res_mol.GetAtoms()))
            ]
            cif_alt_atom_names = [
                res_cif["_chem_comp_atom.alt_atom_id"][i]
                for i in range(len(res_mol.GetAtoms()))
            ]
            real_atom_names = [atom.name for atom in residue]
            atoms = [atom for atom in residue]
            real_atom_coords = []
            for n in cif_atom_names:
                if n in real_atom_names:
                    crd = atoms[real_atom_names.index(n)].get_coord()
                    real_atom_coords.append(crd)

            try:
                coords = np.stack(real_atom_coords, 0)
            except:
                # the atom name could be consistent with alt atom id
                real_atom_coords = []
                for n in cif_alt_atom_names:
                    if n in real_atom_names:
                        crd = atoms[real_atom_names.index(n)].get_coord()
                        real_atom_coords.append(crd)
                coords = np.stack(real_atom_coords, 0)

            distance = np.sqrt(
                np.sum(
                    (coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1
                )
            )
            edge_index = []
            edge_attr = []
            for i in range(distance.shape[0]):
                for j in range(distance.shape[1]):
                    edge_index.append([i, j])
                    edge_attr.append(
                        distance[i][j]
                    )  # edge attr is the distance between 2 residues
            edge_attr = torch.from_numpy(np.array(edge_attr)).float()[..., None]
            edge_index = torch.from_numpy(np.array(edge_index).T).float()
            res_graph.edge_index_fc = edge_index
            res_graph.edge_attr_fc = edge_attr

            # type 1 edge: from smiles (chemical edge)
            # type 2 edge: from fully connect graph (all edges)
            # add an attribute to tell different kind of edges (1 or 2)
            edge_tp1 = torch.zeros((res_graph.edge_attr.shape[0], 1))
            edge_attr1 = torch.cat([res_graph.edge_attr, edge_tp1], -1)
            edge_tp2 = torch.ones((res_graph.edge_attr_fc.shape[0], 1))
            edge_attr2 = torch.cat([res_graph.edge_attr_fc, edge_tp2], -1)
            res_graph.edge_attr = edge_attr1
            res_graph.edge_attr_fc = edge_attr2
            graphs.append(res_graph)

        else:
            # missing residues, these residues do not have fully connect edges
            res_name = res_flag.name
            res_graph, res_mol = from_smiles(
                f"raw_data/residues_lib/{res_name}_ideal.sdf"
            )
            res_graph.edge_index_fc = torch.empty(2, 0)
            res_graph.edge_attr_fc = torch.empty(0, 2)
            edge_tp1 = torch.zeros((res_graph.edge_attr.shape[0], 1))
            edge_attr1 = torch.cat([res_graph.edge_attr, edge_tp1], -1)
            res_graph.edge_attr = edge_attr1
            graphs.append(res_graph)

    return graphs


def single_data_preprocess(data, raw_dir, processed_dir, args):
    """
    Contruct a multi-level graph and surface from peptide and protein sides,
    every simple function to make the data
    """
    data_path = osp.join(processed_dir, data)
    if not osp.exists(data_path):
        os.makedirs(data_path)
    
    pdb_id, chain, target = data.split("_")
    cif_file = osp.join(raw_dir, f"{pdb_id}.cif")
    peptide_info = mmcif_to_graph(cif_file, data, chain)
    protein_graph = mmcif_to_surface(args, cif_file, data, target, data_path)
    torch.save(protein_graph, osp.join(data_path, f"{data}_protein_graph.pt"))
    torch.save(peptide_info, osp.join(data_path, f"{data}_peptide_info.pt"))
    pos, neg, pep_coords = generate_gt(peptide_info, protein_graph, chain, target)
    label = (pos, neg)
    res_graphs = construct_graph(peptide_info, chain)
    ret = PeptideGraph(
        data_name=data,
        peptide_edge_index=peptide_info.edge_index_reindex,
        residue_graph=res_graphs,
    )

    torch.save(ret, osp.join(data_path, f"{data}_peptide_graph.pt"))
    torch.save(label, osp.join(data_path, f"{data}_label.pt"))
    torch.save(pep_coords, osp.join(data_path, f"{data}_peptide_coords.pt"))