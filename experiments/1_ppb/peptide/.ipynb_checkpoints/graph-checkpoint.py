# -*- coding: utf-8 -*-
"""
@File   :  protein.py
@Time   :  2024/01/29 22:28
@Author :  Yufan Liu
@Desc   :  Utilities for pdb/mmcif file process
"""

import dataclasses
from typing import Mapping

import numpy as np
import torch
from Bio.SeqUtils import IUPACData
from scipy.spatial import distance

from peptide.mmcif_parsing import ChainId, PdbStructure, ResidueAtPosition
from peptide.mmcif_parsing import parse as mmcif_parse

PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]
IONS = [
    "CA",
    "Ca",
    "MG",
    "Mg",
    "Zn",
    "ZN",
    "Na",
    "NA",
    "K",
    "FE2",
    "SM",
    "NI",
    "CD",
    "FE",
    "CL",
    "MN",
    "RB",
    "CU",
    "AG",
    "CO",
    "LI",
    "CU1",
    "CU2",
    "H",
    "YB",
    "CS",
    "I",
    "IOD",
    "HG",
    "BA",
    "SR",
    "GD",
    "GA",
    "PR",
    "PB",
    "YT3",
    "TB",
    "O",
    "RE",
    "PT",
    "BR",
    "U1",
    "LA",
    "LU",
    "YB2",
    "ARS",
]

ATOM_VOCAB = {
    "C": 0,
    "O": 1,
    "N": 2,
    "H": 3,
    "S": 4,
    "P": 5,
    "F": 6,
    "Cl": 7,
    "Br": 8,
    "I": 9,
    "Se": 10,
    "B": 11,
    "UNK": 12,
}


@dataclasses.dataclass(frozen=True, repr=True)
class PeptideChain:
    mmcif_structure: PdbStructure
    mmcif_seqres_struct: Mapping[ChainId, Mapping[int, ResidueAtPosition]]
    edge_index_raw: np.ndarray
    edge_index_reindex: np.ndarray
    residue_number_reindex: list
    residue_name: list


def non_standard_connect(cif_data, chain):
    """
    extract residue connections in _struct_conn fileds
    always thinks the direction is from C donor to N donor
    """

    if not "_struct_conn.ptnr1_auth_asym_id" in cif_data.raw_string:
        return False, False

    res_at_position = cif_data.seqres_to_structure[chain]
    ks = [x for x in res_at_position if res_at_position[x].position]
    vs = [res_at_position[x].position.residue_number for x in res_at_position if res_at_position[x].position]
    auth_label_idmap = {str(v): str(k) for k, v in zip(ks, vs)}

    valid_res = []
    for i in res_at_position:
        if res_at_position[i].position:
            valid_res.append(res_at_position[i].position.residue_number)
    valid_res = np.array(valid_res).astype(str)
    cif_dict = cif_data.raw_string
    select_chain = np.where(
        (np.array(cif_dict["_struct_conn.ptnr1_auth_asym_id"]) == chain)
        & (np.isin(np.array(cif_dict["_struct_conn.ptnr1_auth_seq_id"]), valid_res))
        & (np.isin(np.array(cif_dict["_struct_conn.ptnr2_auth_seq_id"]), valid_res))
    )
    if not select_chain:
        raise KeyError(f"Empty chain {chain}")

    connect_dict = {
        "connect_type": np.array(cif_dict["_struct_conn.conn_type_id"])[select_chain],
        "strand_id": np.array(cif_dict["_struct_conn.ptnr1_auth_asym_id"])[select_chain],
        "ptr1_residue": np.array(cif_dict["_struct_conn.ptnr1_auth_comp_id"])[select_chain],
        "ptr2_residue": np.array(cif_dict["_struct_conn.ptnr2_auth_comp_id"])[select_chain],
        "ptr1_residue_id": np.array(cif_dict["_struct_conn.ptnr1_auth_seq_id"])[select_chain],
        "ptr2_residue_id": np.array(cif_dict["_struct_conn.ptnr2_auth_seq_id"])[select_chain],
        "ptr1_atom": np.array(cif_dict["_struct_conn.ptnr1_label_atom_id"])[select_chain],
        "ptr2_atom": np.array(cif_dict["_struct_conn.ptnr2_label_atom_id"])[select_chain],
    }

    src_res = []
    tgt_res = []
    for i in range(len(connect_dict["strand_id"])):
        if connect_dict["connect_type"][i] == "covale":
            # directed for covalent bonds, source C to target N
            if connect_dict["ptr1_atom"][i][0] == "C" and connect_dict["ptr2_atom"][i][0] == "N":
                src_res.append(connect_dict["ptr1_residue_id"][i])
                tgt_res.append(connect_dict["ptr2_residue_id"][i])
            elif connect_dict["ptr1_atom"][i][0] == "N" and connect_dict["ptr2_atom"][i][0] == "C":
                src_res.append(connect_dict["ptr2_residue_id"][i])
                tgt_res.append(connect_dict["ptr1_residue_id"][i])
            else:
                src_res.append(connect_dict["ptr2_residue_id"][i])
                tgt_res.append(connect_dict["ptr1_residue_id"][i])
                src_res.append(connect_dict["ptr1_residue_id"][i])
                tgt_res.append(connect_dict["ptr2_residue_id"][i])
        elif connect_dict["connect_type"][i] == "disulf":
            src_res.append(connect_dict["ptr2_residue_id"][i])
            tgt_res.append(connect_dict["ptr1_residue_id"][i])
            src_res.append(connect_dict["ptr1_residue_id"][i])
            tgt_res.append(connect_dict["ptr2_residue_id"][i])

    src_res = [auth_label_idmap[x] for x in src_res]
    tgt_res = [auth_label_idmap[x] for x in tgt_res]
    edge_index = np.stack([src_res, tgt_res])
    assert edge_index.shape[0] == 2
    return edge_index.astype(int), True


def peptide_bond_from_distance(cif_data, chain_id):
    """
    identify canonical peptide bond from C and N distance
    threshold: 1.5 Angstrom  # usually 1.33-1.34 but nonbond dist will much larger than this
    extract coordinates of C and N in selected chain
    """
    res_at_position = cif_data.seqres_to_structure[chain_id]
    valid_res = []
    for i in sorted(res_at_position.keys()):
        res_flag = res_at_position[i]
        if res_flag.position:
            valid_res.append(
                (
                    res_flag.hetflag,
                    res_flag.position.residue_number,
                    res_flag.position.insertion_code,
                )
            )
        else:
            valid_res.append(None)

    coords_C = []
    residue_id_C = []
    coords_N = []
    residue_id_N = []
    for chain in cif_data.structure:
        if chain.get_id() == chain_id:
            residue_names = [resi.get_id() for resi in chain]
            residues = [resi for resi in chain]

    missing_link = []
    for idx, res in enumerate(valid_res):
        if res in residue_names:
            for atom in residues[residue_names.index(res)]:
                if atom.get_id() == "C":
                    residue_id_C.append(idx)
                    coords_C.append(atom.get_coord())
                elif atom.get_id() == "N":
                    residue_id_N.append(idx)
                    coords_N.append(atom.get_coord())
        else:
            if idx == 0:
                missing_link.append([idx, idx + 1])
            elif idx == len(valid_res) - 1:
                missing_link.append([idx - 1, idx])
            else:
                missing_link.append([idx - 1, idx])
                missing_link.append([idx, idx + 1])

    coords_C = np.stack(coords_C, axis=0)
    coords_N = np.stack(coords_N, axis=0)
    assert coords_C.shape[1] == 3
    dist = distance.cdist(coords_C, coords_N)  # maybe not squared
    ppbonds = np.stack(np.where(dist < 1.5), axis=1)

    edge_index = [(residue_id_C[p[0]], residue_id_N[p[1]]) for p in ppbonds]
    edge_index = np.array(edge_index).T
    if missing_link != []:
        missing_link = np.array(missing_link).T
        edge_index = np.concatenate([edge_index, missing_link], axis=1)
    return edge_index


def mmcif_to_graph(mmcif_path, file_id, chain):
    """Generate peptide graph information from mmcif-format file

    Args:
        mmcif_path: path for cif file
        file_id:
        chain:

    Returns:
        The return class is NOT a graph, but the information of residues,
        how they connected, which is important to generate a residue graph
    """

    with open(mmcif_path, "r") as f:
        cif_data = mmcif_parse(file_id=file_id, mmcif_string="".join(f.readlines()))
    cif_data = cif_data.mmcif_object
    ppbond_index = peptide_bond_from_distance(cif_data, chain)
    non_ppbond_index, non_pp_flag = non_standard_connect(cif_data, chain)

    if non_pp_flag:
        edge_index = np.concatenate([ppbond_index, non_ppbond_index], axis=1)
    else:
        edge_index = ppbond_index
    
    # remove duplicates
    edge_index = np.unique(edge_index, axis=1)
    _, edge_reindex = np.unique(edge_index.flatten(), return_inverse=True)
    edge_reindex = edge_reindex.reshape(edge_index.shape)

    residue_number = [x for x in cif_data.seqres_to_structure[chain] if cif_data.seqres_to_structure[chain][x].position]
    residue_name = [
        cif_data.seqres_to_structure[chain][x].name
        for x in cif_data.seqres_to_structure[chain]
        if cif_data.seqres_to_structure[chain][x].position
    ]
    print(edge_index.max(), edge_index.min(), residue_name, len(residue_name))
    exit()
    peptide_data = PeptideChain(
        mmcif_structure=cif_data.structure,
        mmcif_seqres_struct=cif_data.seqres_to_structure,
        edge_index_raw=torch.from_numpy(edge_index).to(torch.float32),
        edge_index_reindex=torch.from_numpy(edge_reindex).to(torch.float32),
        residue_name=residue_name,
        residue_number_reindex=residue_number,
    )

    return peptide_data
