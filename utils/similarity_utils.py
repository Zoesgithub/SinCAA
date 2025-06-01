from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from openfold.utils.rigid_utils import Rigid
import torch
from scipy.optimize import linear_sum_assignment
import os
import random
import hashlib
import pickle
import time
from sklearn.neighbors import BallTree
periodictable = Chem.GetPeriodicTable()
aa_pattern = "[N!H0]-[C:1]-[C:2](=[O:3])-[OH]"
#num_confs = 20


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

def add_mol_to_emol(emol, mol, exclude_atoms):
    map_from_ori_idx_to_new_idx = {}
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        if atom.GetIdx() not in exclude_atoms:
            map_from_ori_idx_to_new_idx[atom.GetIdx()] = emol.AddAtom(atom)
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if start in map_from_ori_idx_to_new_idx and end in map_from_ori_idx_to_new_idx:
            emol.AddBond(
                map_from_ori_idx_to_new_idx[start], map_from_ori_idx_to_new_idx[end], order=bond.GetBondType())
    return map_from_ori_idx_to_new_idx

def get_idx_of_H(atom):
    neighbors = atom.GetNeighbors()
    for v in neighbors:
        if v.GetSymbol() == "H":
            return v.GetIdx()
        
def myHash(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sample_normed_pos(mid, num_confs, grid_length=1, usecache=True, return_res=True):
    cache_path=os.path.join("tmp/cache/", f"{myHash(mid)}_{num_confs}")
    if os.path.exists(cache_path) and usecache:
        try:
            with open(cache_path, 'rb') as f:
                ret = pickle.load(f)      
            if return_res: 
                return [get_non_empty_coord(_,  ret[1], grid_length)  for _ in ret[0]]
            return
        
        except:
            print("remaking", cache_path)
    
    start="NCC(=O)O"
    end="NCC(=O)O"
    
    start_mol = Chem.AddHs(Chem.MolFromSmiles(start))
    mid_mol = Chem.AddHs(Chem.MolFromSmiles(mid))
    end_mol = Chem.AddHs(Chem.MolFromSmiles(end))

    start_match_pattern = start_mol.GetSubstructMatches(
        Chem.MolFromSmarts(aa_pattern))
    mid_match_pattern = mid_mol.GetSubstructMatches(
        Chem.MolFromSmarts(aa_pattern))
    end_match_pattern = end_mol.GetSubstructMatches(
        Chem.MolFromSmarts(aa_pattern))

    assert len(start_match_pattern) > 0, start_match_pattern
    assert len(mid_match_pattern) > 0, mid_match_pattern
    assert len(end_match_pattern) > 0, end_match_pattern

    start_match_pattern = random.choice(start_match_pattern)
    mid_match_pattern = random.choice(mid_match_pattern)
    end_match_pattern = random.choice(end_match_pattern)

    emol = Chem.EditableMol(Chem.Mol())
    startN, startCA, startCO, _, startOH = start_match_pattern
    midN, midCA, midCO, _, midOH = mid_match_pattern
    endN, endCA, endCO, _, endOH = end_match_pattern

    start_exclude_atoms = [get_idx_of_H(
        start_mol.GetAtomWithIdx(startOH)), startOH]
    start_map = add_mol_to_emol(emol, start_mol, start_exclude_atoms)

    mid_exclude_atoms = [get_idx_of_H(mid_mol.GetAtomWithIdx(
        midOH)), get_idx_of_H(mid_mol.GetAtomWithIdx(midN)), midOH]
    mid_map = add_mol_to_emol(emol, mid_mol, mid_exclude_atoms)

    end_exclude_atoms = [get_idx_of_H(end_mol.GetAtomWithIdx(endN))]
    end_map = add_mol_to_emol(emol, end_mol, end_exclude_atoms)

    b1 = emol.AddBond(
        start_map[startCO], mid_map[midN], order=Chem.rdchem.BondType.SINGLE)
    b2 = emol.AddBond(
        mid_map[midCO], end_map[endN], order=Chem.rdchem.BondType.SINGLE)

    mol = emol.GetMol()
    
    Chem.SanitizeMol(mol)
    cids = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs,  useExpTorsionAnglePrefs=True, useBasicKnowledge=True,  randomSeed=42)
    
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    # build return
    ret_positions=[]
    atom_order={v:i for i,v in enumerate(mid_map.keys())}
    atom_properties=[]
    for cidx, cid in enumerate(cids):
        conf = mol.GetConformer(cid)
        N, CA, CO =atom_order[midN], atom_order[midCA], atom_order[midCO]
        assert mol.GetAtomWithIdx(mid_map[midN]).GetAtomicNum()==7
        assert mol.GetAtomWithIdx(mid_map[midCA]).GetAtomicNum()==6
        assert mol.GetAtomWithIdx(mid_map[midCO]).GetAtomicNum()==6
        positions = np.zeros((len(atom_order), 3))
        for i,v in mid_map.items():
            positions[atom_order[i]]=conf.GetAtomPosition(v)
            if cidx==0:
                atom=mol.GetAtomWithIdx(v)
                radius = periodictable.GetRvdw(atom.GetAtomicNum())
                atom_num = atom.GetAtomicNum()
                charge = float(atom.GetProp("_GasteigerCharge"))
                if np.isnan(charge):
                    charge = 0
                atom_properties.append([float(radius), atom_num, charge])
        ret_positions.append(normalize_positions(
            np.array(positions), N, CA, CO))
    ret=(ret_positions, np.array(atom_properties), grid_length)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(ret, f)
    except:
        print("fail to save", cache_path)
    
    if return_res:
        return  [get_non_empty_coord(_,  ret[1], grid_length)  for _ in ret[0]]
    return
    
    


def get_non_empty_coord(position, atom_properties, grid_length):
    atom_radius=atom_properties[..., :1]
    atom_props = atom_properties[..., 1:]
    min_pos = (position-atom_radius/2)/grid_length
    max_pos = (position+atom_radius/2)/grid_length+1
    
    ret_res = {}
    for prop, mipv, mapv in zip(atom_props, min_pos, max_pos):
        min_x, min_y, min_z = mipv
        max_x, max_y, max_z = mapv
        for i in range(int(min_x), int(max_x)):
            for j in range(int(min_y), int(max_y)):
                for k in range(int(min_z), int(max_z)):
                    if (i, j, k) not in ret_res:
                        ret_res[(i, j, k)] = {}
                    num=int(prop[0])
                    if num not in ret_res[(i, j, k)]:
                        ret_res[(i, j, k)][num]=0
                    ret_res[(i, j, k)][num]+=prop[1]
                    
    return ret_res


def compute_pair(grid_a, grid_b):
    overlap = 0
    overlap_keys=set(grid_a.keys()).intersection(grid_b.keys())
    for k in overlap_keys:
        info_a = grid_a[k]
        info_b = grid_b[k]
        overk = set(info_a.keys()).intersection(info_b.keys())
        value = 0
        if len(overk)>0:
            for k in overk:
                value += max(1-abs(info_a[k]-info_a[k]), 0)
        overlap += value/max(len(overk), 1)
        assert not np.isnan(
            overlap), f"{overlap} {value} {len(overk)}"
    num_a = len(grid_a)
    num_b = len(grid_b)
    ret =float(overlap*1.0/max(num_a+num_b-overlap, 1))
    assert not np.isnan(ret), f"{grid_a} {grid_b}"

    return ret

def count_distance(a, b, threshold=-1):

    distance_matrix = np.zeros([len(a), len(b)])-1
    if threshold > -1:
        max_v=0
        for i,pa in enumerate(a):
            pb=random.choice(b)
            d=compute_pair(pa, pb)
            distance_matrix[i, pb]=d
            max_v=max(max_v, d)
        if max_v<threshold:
            return 0
    
    for i, pa in enumerate(a):
        for j, pb in enumerate(b):
            if distance_matrix[i,j]<0:
                d = compute_pair(pa, pb)
                distance_matrix[i, j] = d
    
    row_ind, col_ind = linear_sum_assignment(1-distance_matrix)
    distance = distance_matrix[row_ind, col_ind]
    assert distance.shape[0]>0
    return distance.sum()/distance.shape[0]

def myHash(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_space_distance(a, b, grid_length=0.5):
    a_pos = sample_normed_pos(a,20, grid_length=grid_length)
    b_pos = sample_normed_pos(b,20, grid_length=grid_length)
    #if not isinstance(a_pos, int) and not isinstance(b_pos, int):
    ret = count_distance(a_pos, b_pos)
    return ret
