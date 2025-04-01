import numpy as np
from rdkit import Chem
from utils.data_constants import SmartsOfRotableBond, SymetricTorsionAngleRmsdThreshold, cache_data_path
import rdkit.Chem.rdMolTransforms as rdmt
import networkx as nx
import os
import pickle
from utils.feats_utils import get_torsion_angle_from_four_points
import torch

def kabsch_rotation(P, Q): # P is target, Q is ref
    C = P.transpose(-1, -2) @ Q
    V, _, W = torch.linalg.svd(C)
    d = (torch.linalg.det(V) * torch.linalg.det(W)) < 0.0
    if d:
        mask=V.new_ones(V.shape)
        mask[:, -1]*=-1
        V=V*mask
        #V[:, -1] = -V[:, -1]
    U = V @ W
    assert not torch.isnan(U.sum()), f"{P} {Q}"
    return U


def get_optimal_transform(src_atoms, tgt_atoms):
    src_center = src_atoms.mean(-2)[None, :]
    tgt_center = tgt_atoms.mean(-2)[None, :]
    r = kabsch_rotation(src_atoms - src_center, tgt_atoms - tgt_center)
    x = tgt_center - src_center @ r
    return r, x


def compute_rmsd(true_atom_pos, pred_atom_pos):
    assert true_atom_pos.shape==pred_atom_pos.shape
    sd_square = ((true_atom_pos - pred_atom_pos)**2).sum(axis=-1).mean().clamp(1e-6).sqrt()
    return sd_square


def get_optimal_align(target, ref):
    optimal_rot, optimal_trans = get_optimal_transform(target, ref)
    rmsd_square = compute_rmsd(ref, target@optimal_rot+optimal_trans)
    return rmsd_square

def get_optimal_align_by_residue(target, ref, node_res_idx, eps=1e-6):
    batch_size=node_res_idx.max()+1
    target_center=torch.scatter_reduce(target.new_zeros([batch_size, target.shape[-1]]), 0, node_res_idx[..., None].expand_as(target),target,  reduce="mean", include_self=False)
    ref_center=torch.scatter_reduce(ref.new_zeros([batch_size, ref.shape[-1]]), 0, node_res_idx[..., None].expand_as(ref),ref, reduce="mean", include_self=False)
    
    centered_target=target-target_center[node_res_idx]
    centered_ref=ref-ref_center[node_res_idx]
    Cs=[]
    for i in range(node_res_idx.max()+1):
        select_idx=node_res_idx==i
        Cs.append(centered_target[select_idx].transpose(-1, -2)@centered_ref[select_idx])
    C=torch.stack(Cs, 0)
    V, _, W = torch.linalg.svd(C)
    d = (torch.linalg.det(V) * torch.linalg.det(W)) < 0.0

    mask=V.new_ones(V.shape)
    mask[d, :, -1]*=-1
    V=V*mask

    optimal_rot =torch.matmul(V , W)
    optimal_trans = ref_center - torch.matmul(target_center[:, None, :], optimal_rot).squeeze(-2)
    rmsd_square = ((ref-(torch.matmul(target[:, None, :],optimal_rot[node_res_idx]).squeeze(-2)+optimal_trans[node_res_idx]))**2).sum(-1).add(eps)
    rmsd_square=torch.scatter_reduce(rmsd_square.new_zeros(batch_size), 0, node_res_idx, rmsd_square, reduce="mean", include_self=False)
    return rmsd_square.add(eps).sqrt()

def mol_to_graph(mol):
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        graph.add_node(atom_idx, symbol=atom_symbol)

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = str(bond.GetBondType())
        graph.add_edge(begin_idx, end_idx,
                       bond_type=bond_type)
    return graph


def get_atom_groups(mol, exist_atoms): # make position embedding of equi atoms
    mol=Chem.Mol(mol) # make a copy of mol
    self_match = mol.GetSubstructMatches(mol, uniquify=False)
    
    # unify matches
    uni_match = set()
    num_atom=len(mol.GetAtoms())
    for v in self_match:
        v = tuple(_  if mol.GetAtomWithIdx(_).GetSymbol() != "H" else -99999 for _ in v)
        uni_match.add(v)
    uni_match = [list(_) for _ in uni_match]
    def count_conflict(x):
        ret=0
        for i in range(len(x)-1):
            if x[i]>x[i+1]:
                ret+=1
        return ret
    uni_match=sorted(uni_match, key=lambda x:count_conflict(x)) # each row of uni match contains equi atoms
    
    # generate position embedding
    ret=[-1 for _ in range(num_atom)]
    grouped_atoms={}
    handled_atoms=set()
    for i in range(num_atom):
        grouped_atoms[i]=set()
        for v in uni_match:
            if v[i] not in handled_atoms and v[i] in exist_atoms:
                handled_atoms.add(v[i])
                grouped_atoms[i].add(v[i])
                ret[v[i]]=len(grouped_atoms[i])
    if len(uni_match)>0:
        uni_match=np.array(uni_match).transpose(1, 0)
    else:
        uni_match=np.array(uni_match).reshape(-1, num_atom).transpose(1, 0)
    return ret, uni_match
