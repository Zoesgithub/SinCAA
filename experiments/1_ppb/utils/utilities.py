# -*- coding: utf-8 -*-
"""
@File   :  utilities.py
@Time   :  2024/02/03 14:50
@Author :  Yufan Liu
@Desc   :  auxliliary functions
"""

import networkx as nx
import numpy as np
import psutil
from Bio.PDB import PDBIO, Atom, Chain, Model, Residue, Structure


def draw_directed_graph(chain_seq_num, chain_residue, edge_index):

    assert edge_index.shape[0] == 2
    labels = dict(zip(chain_seq_num.astype(int) - 1, chain_residue))

    idx = []
    for point in np.array(list(map(tuple, edge_index.T))):
        idx.append((point[0], point[1]))
    G = nx.DiGraph()
    G.add_edges_from(idx)

    nx.draw_networkx(G, pos=nx.spring_layout(G), labels=labels, with_labels=True)


def write_to_pointcloud(coordinates, bfactors, output):
    """
    Write coordinates into a point clould PDB file, using biopython
    coordinates: ndarray of shape [N, 3]
    bfactors: ndarray of shape [N, ]
    """

    assert len(coordinates) == len(bfactors), "File length not match."
    structure = Structure.Structure("1")
    model = Model.Model(0)
    structure.add(model)
    chain = Chain.Chain("A")
    model.add(chain)

    for i, bf in enumerate(bfactors):
        res = Residue.Residue((" ", i + 1, " "), "XYZ", "")
        chain.add(res)

        atom = Atom.Atom("H", coordinates[i], bf, 1.0, " ", "H", i + 1, "H")
        res.add(atom)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output)


def cpu_monitor(tp="used"):
    mem = psutil.virtual_memory()
    if tp == "total":
        return mem.total / (1024**3)
    elif tp == "used":
        return mem.used / (1024**3)


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size_mb = total_params * 4 / (1024 * 1024)  # Assuming 4 bytes per parameter
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Total size of trainable parameters: {total_size_mb:.2f} MB")
    return total_size_mb
