import numpy as np
from rdkit import Chem
from peptide.amino_acid import allowable_features, MolBondDict, PeriodicTable

add_node_num_feats = [
    "edges", "defined_torsion_angle", "pseudo_backbone_atoms", "nodes_reindex_when_alt_torsion_angles", "local_frame_indicator", "other_edges", "last_atom_idx","default_CA"]

BondThreshold=1.5
def build_conn_from_distance(init_graph, append_graph=None, connections=None, default_CO_and_N=None, amino_acid_seq=None):
    num_nodes = len(init_graph["nodes_int_feats"])
    same_graph = False
    if append_graph is None:
        append_graph = init_graph
        same_graph = True

    pair_gt_exists_mask = append_graph["gt_position_exists"][...,
                                                             None]*init_graph["gt_position_exists"][..., None, :]

    pair_gt_distance = np.sqrt(
        ((append_graph["gt_positions"][..., None, :]-init_graph["gt_positions"][..., None, :, :])**2).sum(-1))

    init_atom_is_s = (init_graph["nodes_int_feats"][..., 0] == 16)*1.
    append_atom_is_s = (append_graph["nodes_int_feats"][..., 0] == 16)*1.

    if "atom_is_ion" in append_graph:
        append_atom_is_ion = (append_graph["atom_is_ion"])
    else:
        append_atom_is_ion = np.zeros(len(append_graph["nodes_int_feats"]))

    pair_contain_ion = append_atom_is_ion[...,
                                          None]
    pair_gt_exists_mask = pair_gt_exists_mask*(1-pair_contain_ion)

    cross_edges = []
    cross_edge_attrs = []
    edge_length = []

    assert len(
        pair_gt_distance.shape) == 2 and pair_gt_distance.shape == pair_gt_exists_mask.shape

    has_extra_bond = (pair_gt_distance <BondThreshold) & (
        pair_gt_exists_mask > 0) & (pair_gt_distance > 0)
    if same_graph:
        has_extra_bond = has_extra_bond & (abs(
            append_graph["nodes_residue_index"][..., None]-init_graph["nodes_residue_index"][..., None, :]) > 0)

    SSbond = (pair_gt_distance >= BondThreshold) & (pair_gt_exists_mask > 0) & (
        pair_gt_distance < 2.1) & ((init_atom_is_s[..., None, :]*append_atom_is_s[..., None]) == 1)

    has_extra_bond_position = np.nonzero(np.array(has_extra_bond) | np.array(SSbond))

    predefine_conn = None
    if connections is not None:
        assert same_graph
        predefine_conn = {}
        for a in connections:
            for b in connections[a]:
                if isinstance(b, int):
                    start, end = connections[a][b]
                    start = connections[a][start]
                    end = connections[b][end]
                    if start not in predefine_conn:
                        predefine_conn[start] = set()
                    predefine_conn[start].add(end)

    if default_CO_and_N is not None:
        assert same_graph
        assert amino_acid_seq is not None
        for i in default_CO_and_N.keys():
            if i in default_CO_and_N and i-1 in default_CO_and_N:
                cur_co, cur_n = default_CO_and_N[i]
                prev_co, prev_n = default_CO_and_N[i-1]
                if prev_co is not None and (amino_acid_seq[i].is_missing or amino_acid_seq[i-1].is_missing):
                    cross_edges.append([prev_co, cur_n])
                    cross_edges.append([cur_n, prev_co])
                    if predefine_conn is not None:
                        if prev_co in predefine_conn and cur_n in predefine_conn[prev_co]:
                            predefine_conn[prev_co].remove(cur_n)
                        if cur_n in predefine_conn and prev_co in predefine_conn[cur_n]:
                            predefine_conn[cur_n].remove(prev_co)
                    cross_edge_attrs.append([MolBondDict["SINGLE"], allowable_features['possible_bond_dirs'].index(Chem.rdchem.BondDir.NONE)])
                    cross_edge_attrs.append([MolBondDict["SINGLE"], allowable_features['possible_bond_dirs'].index(Chem.rdchem.BondDir.NONE)])
                    
                    edge_length.append(0)
                    edge_length.append(0)


    if predefine_conn is not None:
        for k in predefine_conn:
            for v in predefine_conn[k]:
                cross_edges.append([k, v])
                cross_edge_attrs.append([MolBondDict["SINGLE"], allowable_features['possible_bond_dirs'].index(Chem.rdchem.BondDir.NONE)])
                edge_length.append(0)
    if same_graph:
        if len(has_extra_bond_position)>0:
            for p1, p2 in zip(*has_extra_bond_position):
                edge_length.append(pair_gt_distance[p1, p2])
                cross_edges.append([p1, p2])
                cross_edge_attrs.append([MolBondDict["SINGLE"], allowable_features['possible_bond_dirs'].index(Chem.rdchem.BondDir.NONE)])
    elif len(has_extra_bond_position)>0:
        for p1, p2 in zip(*has_extra_bond_position):
            edge_length.append(pair_gt_distance[p1, p2])
            edge_length.append(pair_gt_distance[p1, p2])
            cross_edges.append([p1+num_nodes, p2])
            cross_edges.append([p2, p1+num_nodes])
            cross_edge_attrs.append([MolBondDict["SINGLE"], allowable_features['possible_bond_dirs'].index(Chem.rdchem.BondDir.NONE)])
            cross_edge_attrs.append([MolBondDict["SINGLE"], allowable_features['possible_bond_dirs'].index(Chem.rdchem.BondDir.NONE)])
    return {
        "edge_lengths": np.array(edge_length),
        "edges": np.array(cross_edges).reshape(-1, 2),
        "edge_attrs": np.array(cross_edge_attrs).reshape(-1, 2)
    }
