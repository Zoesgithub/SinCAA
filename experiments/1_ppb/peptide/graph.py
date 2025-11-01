# -*- coding: utf-8 -*-

import dataclasses
from typing import Mapping

import numpy as np
import torch
from Bio.SeqUtils import IUPACData
from scipy.spatial import distance

from peptide.mmcif_parsing import ChainId, PdbStructure, ResidueAtPosition
from peptide.mmcif_parsing import parse as mmcif_parse
from peptide.chain import Chain

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
    embs: np.ndarray
    pseudo_emb: np.ndarray
    cords: np.ndarray
    cord_exists: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    atom_types: np.ndarray
    residue_index: np.ndarray
    residue_name: list
    atom_name: list
    mmcif_structure: PdbStructure
    mmcif_seqres_struct: Mapping[ChainId, Mapping[int, ResidueAtPosition]]
    
@dataclasses.dataclass(frozen=True, repr=True) 
class SPeptideChain:
    embs: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    atom_types: np.ndarray
    residue_index: np.ndarray
    cords:np.ndarray
    
@dataclasses.dataclass(frozen=True, repr=True) 
class HPeptideChain:
    embs: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    atom_types: np.ndarray
    residue_index: np.ndarray
    cords:np.ndarray
    label_residue_index: np.ndarray
    

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


def mmcif_to_graph(mmcif_path, file_id, chain, emb_path, pretrained_sincaa):
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
    chain_graph=Chain(chain, cif_data.seqres_to_structure[chain], cif_data.chain_to_seqres[chain], cif_data.structure[chain], cif_data, emb_path=emb_path).get_graph_with_gt()
    batch_id=np.zeros(len(chain_graph["nodes_int_feats"]))
    inp_feat={"batch_id":torch.tensor(batch_id).cuda()}
    for k in ["nodes_int_feats", "nodes_float_feats", "edges", "edge_attrs"]:
        inp_feat[k]=torch.tensor(chain_graph[k]).cuda()
    inp_feat["node_residue_index"]=inp_feat["batch_id"]
    pretrained_sincaa,get_emb_from_feat=pretrained_sincaa
    emb=get_emb_from_feat(inp_feat, pretrained_sincaa, inp_feat["batch_id"].device)[0]
    emb=emb.detach().cpu().numpy()
    peptide_data = PeptideChain(
        embs=emb,
        pseudo_emb=emb.mean(0),
        cords=chain_graph["gt_positions"],
        cord_exists=chain_graph["gt_position_exists"],
        edge_index=chain_graph["edges"],
        edge_attr=chain_graph["edge_attrs"],
        atom_types=chain_graph["atom_types"],
        atom_name=chain_graph["atom_names"],
        residue_name=chain_graph["aa_names"],
        residue_index=chain_graph["nodes_residue_index"],
        mmcif_structure=cif_data.structure,
        mmcif_seqres_struct=cif_data.seqres_to_structure,
    )
   
    return peptide_data
    