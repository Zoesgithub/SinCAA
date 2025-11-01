# -*- coding: utf-8 -*-
"""
@File   :  gt.py
@Time   :  2024/02/06 15:00
@Author :  Yufan Liu
@Desc   :  Ground truth and augmentations for training
"""

import numpy as np
from scipy.spatial import distance
from utils.utilities import write_to_pointcloud
import time


def generate_gt(
    peptide_info, protein_graph, chain_id, protein_chain_id, cutoff=2.678, return_dist_only=False
):
    # generate interative pairs of peptide and vertex
    pep_coords=peptide_info.cords
    peptide_position_exists=peptide_info.cord_exists
    
    vertex_coords = protein_graph.mesh_vertex

    dists = distance.cdist(pep_coords, vertex_coords)
    dists[peptide_position_exists<1]=-1
    pep_len, pro_len = dists.shape
    pos = list(
        set(
            [
                (i, j)
                for i in range(pep_len)
                for j in range(pro_len)
                if dists[i, j] < cutoff and dists[i,j]>0 #and i in pep_pos
            ]
        )
    )
    neg = list(set([(i, j) for i in range(pep_len) for j in range(pro_len) if dists[i, j] > cutoff]))
   
    neg = [n for n in neg if n not in pos]

    return pos, neg, pep_coords
