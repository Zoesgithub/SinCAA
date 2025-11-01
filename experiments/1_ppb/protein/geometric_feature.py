# -*- coding: utf-8 -*-
"""
@File   :  geometric_feature.py
@Time   :  2024/02/06 12:58
@Author :  Yufan Liu
@Desc   :  Compute Geometric feature, refer to MaSIF
"""

import numpy as np
from scipy.spatial import distance
import time


def generate_shapeindex(mesh):
    # Gaussian and Mean
    # num_v = mesh.current_mesh().vertex_number()
    # mesh.meshing_repair_non_manifold_vertices()  # remove edges
    _ = mesh.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype="Mean Curvature")
    # this will update vertex scalar
    H = mesh.current_mesh().vertex_scalar_array()
    _ = mesh.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype="Gaussian Curvature")
    K = mesh.current_mesh().vertex_scalar_array()

    elem = np.square(H) - K
    elem[elem < 0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index
    si = (k1 + k2) / (k1 - k2)
    si = np.arctan(si) * (2 / np.pi)
    # assert(len(si) == num_v), print(len(si), num_v)

    mesh.set_attribute("shape_index", si)
    return mesh


def edge_to_neigh(edge_index):
    """
    return all neighbors from edge index
    """
    neighbors = {}
    for u, v in edge_index.T:
        if u not in neighbors:
            neighbors[u] = [u]
        if v not in neighbors:
            neighbors[v] = [v]
        if v not in neighbors[u]:
            neighbors[u].append(v)
        if u not in neighbors[v]:
            neighbors[v].append(u)
    return neighbors


def generate_ddc(args, mesh, edge_index):
    n = len(mesh.vertices)
    normals = mesh.vertex_normals
    input_feat = np.zeros((n, 3))
    shapeindex = mesh.get_attribute("shape_index")
    charge = mesh.get_attribute("atom_type")
    dist = mesh.get_attribute("atom_dist")
    neighbors = edge_to_neigh(edge_index)
    # use the feature
    new_index = []
    ddc_attrs = []
    tc = time.time()
    for vix in range(n):
        neigh_vix = np.array(neighbors[vix])
        # Compute the distance-dependent curvature for all neighbors of the patch.
        patch_v = mesh.vertices[neigh_vix]
        patch_n = normals[neigh_vix]
        patch_cp = np.where(neigh_vix == vix)[0][0]  # central point position
        patch_rho = distance.cdist(patch_v[0][None, ...], patch_v)[0]
        ddc = compute_ddc(patch_v, patch_n, patch_cp, patch_rho)
        for i, j in enumerate(neigh_vix):
            if i != 0:  # now self loops
                new_index.append([neigh_vix[0], j])
                new_index.append([j, neigh_vix[0]])
                ddc_attrs.append(ddc[i])
                ddc_attrs.append(ddc[i])

        input_feat[vix, 0] = shapeindex[neigh_vix][patch_cp]
        input_feat[vix, 1] = charge[neigh_vix][patch_cp]
        input_feat[vix, 2] = dist[neigh_vix][patch_cp]
    new_index = np.array(new_index).T
    ddc_attrs = np.array(ddc_attrs)
    new_index, unique_id = np.unique(new_index, return_index=True, axis=1)
    ddc_attrs = ddc_attrs[unique_id]
    assert new_index.shape == edge_index.shape, print(new_index.shape, edge_index.shape, "Not consistent.")
    mesh.set_attribute("input_feature", input_feat)
    return mesh, new_index, ddc_attrs


def compute_ddc(patch_v, patch_n, patch_cp, patch_rho):
    """
    Compute the distance dependent curvature, Yin et al PNAS 2009
        patch_v: the patch vertices
        patch_n: the patch normals
        patch_cp: the index of the central point of the patch
        patch_rho: the geodesic distance to all members.
    Returns a vector with the ddc for each point in the patch.
    """
    n = patch_n
    r = patch_v
    i = patch_cp
    # Compute the mean normal 2.5A around the center point
    ni = mean_normal_center_patch(patch_rho, n, 2.5)
    dij = np.linalg.norm(r - r[i], axis=1)
    # Compute the step function sf:
    sf = r + n
    sf = sf - (ni + r[i])
    sf = np.linalg.norm(sf, axis=1)
    sf = sf - dij
    sf[sf > 0] = 1
    sf[sf < 0] = -1
    sf[sf == 0] = 0
    # Compute the curvature between i and j
    dij[dij == 0] = 1e-8
    kij = np.divide(np.linalg.norm(n - ni, axis=1), dij)
    kij = np.multiply(sf, kij)
    # Ignore any values greater than 0.7 and any values smaller than 0.7
    kij[kij > 0.7] = 0
    kij[kij < -0.7] = 0
    return kij


def mean_normal_center_patch(D, n, r):
    """
    Function to compute the mean normal of vertices within r radius of the center of the patch.
    """
    c_normal = [n[i] for i in range(len(D)) if D[i] <= r]
    mean_normal = np.mean(c_normal, axis=0, keepdims=True).T
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    return np.squeeze(mean_normal)
