from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from openfold.utils.rigid_utils import Rigid
import torch
from scipy.optimize import linear_sum_assignment
import os
import random
import hashlib
periodictable = Chem.GetPeriodicTable()
aa_pattern = "[N!H0]-[C:1]-[C:2](=[O:3])-[OH]"
num_confs = 20


def normalize_positions(position: np.array, N_pos: int, CA_pos: int, CO_pos: int) -> np.array:
    tensor_pos = torch.tensor(position)
    rigid = Rigid.from_3_points(
        tensor_pos[N_pos], tensor_pos[CA_pos], tensor_pos[CO_pos])
    normed_position = rigid.invert_apply(tensor_pos)
    return normed_position.numpy()


def sample_pos(smiles, prefix, num_confs):
    ret = []
    if not os.path.exists(f'{prefix}_{num_confs-1}.mol'):
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMultipleConfs(
                mol, numConfs=num_confs,  useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
            for conf_id in range(num_confs):
                molBlock = Chem.MolToMolBlock(mol, confId=conf_id)
                with open(f'{prefix}_{conf_id}.mol', 'w') as f:
                    f.write(molBlock)
        except:
            return 1
    for i in range(num_confs):
        ret.append(Chem.MolFromMolFile(
            f'{prefix}_{i}.mol',  sanitize=True, removeHs=False))
    return ret


def sample_normed_pos(smiles: str, prefix, grid_length=1) -> list:
    confs = sample_pos(smiles, prefix, num_confs)

    if isinstance(confs, int):
        return 1
    ret_positions = []
    atom_properties = []
    for mol in confs:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        match_pattern = mol.GetSubstructMatches(Chem.MolFromSmarts(aa_pattern))
        assert len(match_pattern) > 0

        N, CA, CO = match_pattern[0][0], match_pattern[0][1], match_pattern[0][2]
        # get N/CA/C idx

        conf = mol.GetConformer(0)
        positions = []
        for i, atom in enumerate(mol.GetAtoms()):
            positions.append(conf.GetAtomPosition(atom.GetIdx()))
            assert i == atom.GetIdx()
        ret_positions.append(normalize_positions(
            np.array(positions), N, CA, CO))
    for i, atom in enumerate(mol.GetAtoms()):
        radius = periodictable.GetRvdw(atom.GetAtomicNum())
        atom_num = atom.GetAtomicNum()
        charge = float(atom.GetProp("_GasteigerCharge"))
        if np.isnan(charge):
            charge = 0
        atom_properties.append([float(radius), atom_num, charge])

    coord = [get_non_empty_coord(_,  np.array(
        atom_properties), grid_length) for _ in ret_positions]
    return ret_positions, np.array(atom_properties), coord


def get_non_empty_coord(position, atom_properties, grid_length):
    #atom_radius=atom_properties[..., 0]
    atom_props = atom_properties[..., 1:]
    min_pos = (position)/grid_length
    max_pos = (position)/grid_length+1
    ret_res = {}
    for prop, mipv, mapv in zip(atom_props, min_pos, max_pos):
        min_x, min_y, min_z = mipv
        max_x, max_y, max_z = mapv
        for i in range(int(min_x), int(max_x)):
            for j in range(int(min_y), int(max_y)):
                for k in range(int(min_z), int(max_z)):
                    if (i, j, k) not in ret_res:
                        ret_res[(i, j, k)] = []
                    ret_res[(i, j, k)].append(prop)
    return ret_res


def compute_pair(grid_a, grid_b):
    overlap = 0
    def sim_func(a, b):
        ret = -float(a[0] == b[0])+abs(a[1]-b[1])
        assert not np.isnan(ret), f"{a} {b}"
        return ret
    for k in grid_a:
        if k in grid_b:
            info_a = grid_a[k]
            info_b = grid_b[k]
            distance = [[sim_func(a, b) for b in info_b] for a in info_a]
            row_ind, col_ind = linear_sum_assignment(distance)
            info_a = [info_a[_] for _ in row_ind]
            info_b = [info_b[_] for _ in col_ind]
            norm_factor = max(min(len(info_a), len(info_b)), 1)
            value = 0
            for a, b in zip(info_a, info_b):
                if a[0] == b[0]:
                    value += max(1-abs(a[1]-b[1]), 0)
            overlap += value/norm_factor
            assert not np.isnan(
                overlap), f"{overlap} {value} {norm_factor} {a} {b}"
    num_a = len(grid_a)
    num_b = len(grid_b)
    ret =float(overlap*1.0/max(num_a+num_b-overlap, 1))
    assert not np.isnan(ret), f"{grid_a} {grid_b}"
    return ret

def count_random_distance(a,b, sample_num=5):
    try:
        max_v=0
        for i in range(sample_num):
            pa=random.choice(a)
            pb=random.choice(b)
            max_v+= compute_pair(pa, pb)
    except:
        return 0
    return max_v

def count_distance(a, b, threshold=-1):
    distance_matrix = np.zeros([len(a), len(b)])
    if threshold > -1:
        max_v=0
        for i,pa in enumerate(a):
            pb=random.choice(b)
            max_v=max(max_v, compute_pair(pa, pb))
        if max_v<threshold:
            return 0
    for i, pa in enumerate(a):
        for j, pb in enumerate(b):
            d = compute_pair(pa, pb)
            distance_matrix[i, j] = d
    row_ind, col_ind = linear_sum_assignment(1-distance_matrix)

    distance = distance_matrix[row_ind, col_ind]
    return distance.sum()/distance.shape[0]

def myHash(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_space_distance(a, b, grid_length=1):
    a_pos = sample_normed_pos(a, f'data/cache/{myHash(a)}', grid_length=grid_length)
    b_pos = sample_normed_pos(b, f'data/cache/{myHash(b)}', grid_length=grid_length)
    if not isinstance(a_pos, int) and not isinstance(b_pos, int):
        ret = count_distance(a_pos[-1], b_pos[-1])
        return ret
    return None
