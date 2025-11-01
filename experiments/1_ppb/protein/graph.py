# -*- coding: utf-8 -*-
"""
@File   :  surface.py
@Time   :  2024/02/05 15:06
@Author :  Yufan Liu
@Desc   :  Generate protein surface and vertices fingerprints
"""

import numpy as np
from torch_geometric.data import Data

from peptide.mmcif_parsing import parse as mmcif_parse
from Bio.SeqUtils import seq1
from protein.pepnn_compute_features import node_features, edge_features, nearest_neighbors, calc_dihedral_angles, get_nearest_neighbor_distances, rbf, num_rbf, get_rotamer_locations_and_distances, num_rbf_rot, get_orientation_features, positional_embedding, number_pos_encoding
from Bio import PDB

def mmcif_to_prograph(args, mmcif_path, file_id, chain, path):
    """This function generate protein surface from mmcif-formated file

    Args:
        args: from arguments file
        mmcif_path: input mmcif file of protein
        file_id: i.e. the input file name
        chain: receptor chains of the complex
        path: path for store processed files

    Returns:
        A PyG graph of generated surface (mesh)
    """
    process_path = path
    chain_split = chain.split("-")
    with open(mmcif_path, "r") as f:
        cif_data = mmcif_parse(file_id=file_id, mmcif_string="".join(f.readlines()))
    cif_data = cif_data.mmcif_object
    chain_to_seqres=cif_data.chain_to_seqres
    seqres_to_structure=cif_data.seqres_to_structure
    cif_struct = cif_data.structure
    
    # get struc info
    atom_coords = []
    atom_names=[]
    residue_indices = []  
    reindex_residue_indices = []  
    residue_type=[]
    chain_ids=[]
    map_from_chain_id_to_seq={}
    reindex_ri=0
    for  chain_id in  chain_split:
        map_from_chain_id_to_seq[chain_id]=[chain_id, chain_to_seqres[chain_id]]
        s=chain_to_seqres[chain_id]
        chain= seqres_to_structure[chain_id]
        st=cif_struct[chain_id]
        for j in chain:
            residue=chain[j] # zero-base
            assert seq1(residue.name)==s[j], f"{seq1(residue.name)} {s[j]} {j} {s}"
            if not residue.is_missing: 
                structure = st[(residue.hetflag,  residue.position.residue_number, residue.position.insertion_code)]
                for atom in structure:
                    atom_coords.append(atom.get_coord())
                    atom_names.append(atom.name)
                    residue_indices.append(j) 
                    reindex_residue_indices.append(reindex_ri)
                    residue_type.append(residue.name)
                    chain_ids.append(chain_id)
                reindex_ri+=1
    
    atom_coords = np.array(atom_coords)
    residue_indices = np.array(residue_indices)
    reindex_residue_indices=np.array(reindex_residue_indices)
    assert len(atom_coords)>0
    total_res_num=sum([len(_[1]) for _ in map_from_chain_id_to_seq.values()])
    sort_chain_keys=sorted(map_from_chain_id_to_seq.keys())
    # build pepnn feat
    node_feat=np.zeros([total_res_num, node_features])
    edge_feat=np.zeros([total_res_num,nearest_neighbors,  edge_features])
    i=0
    for k in sort_chain_keys:
        s=map_from_chain_id_to_seq[k][1]
        for t in s:
            if t!="X":
                node_feat[i, PDB.Polypeptide.d1_to_index[t]]=1
            i+=1
    angles = calc_dihedral_angles(np.array(atom_names), atom_coords, np.array(chain_ids), sort_chain_keys, residue_indices, map_from_chain_id_to_seq) 
    node_feat[:, 20:26] = angles 
    neighbor_distances, neighbor_indices= get_nearest_neighbor_distances(
        np.array(atom_names), atom_coords, reindex_residue_indices, np.array(chain_ids), sort_chain_keys, residue_indices, map_from_chain_id_to_seq)
    adjusted_neighbor_distances = rbf(neighbor_distances)
    edge_feat[:, :, 0:num_rbf] = adjusted_neighbor_distances
    rotamer_locations, rotamer_distances, rotamer_ca_cord = get_rotamer_locations_and_distances(np.array(atom_names), atom_coords,residue_indices, reindex_residue_indices, np.array(chain_ids), sort_chain_keys, map_from_chain_id_to_seq)
    adjusted_rotamer_distances = rbf(np.array(rotamer_distances),
                                           True)
    node_feat[:, 26:26+num_rbf_rot] = adjusted_rotamer_distances
    neighbor_directions, neighbor_orientations, rotamer_directions = get_orientation_features(
            rotamer_ca_cord, neighbor_indices, rotamer_locations, sort_chain_keys, map_from_chain_id_to_seq)
    
    edge_feat[:, :, num_rbf:num_rbf+3] = neighbor_directions
    edge_feat[:, :, num_rbf+3:num_rbf+7] = neighbor_orientations
    node_feat[:, 26+num_rbf_rot:26+num_rbf_rot+3] = rotamer_directions
    
    edge_embeddings = positional_embedding(neighbor_indices)

    edge_feat[:, :, num_rbf+7: num_rbf+7+number_pos_encoding] = edge_embeddings
    # save
    protein_info=Data(x=atom_coords,node_feat=node_feat,neighbor_indices=neighbor_indices, edge_feat=edge_feat, reindex_residue_indices=reindex_residue_indices, residue_indices=residue_indices, chain_ids=chain_ids, map_from_chain_id_to_seq=map_from_chain_id_to_seq, atom_names=atom_names, residue_type=residue_type)
    return protein_info
