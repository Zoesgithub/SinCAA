from torch.utils.data import Dataset
import pandas as pd
from utils.similarity_utils import aa_pattern, num_confs
from rdkit import Chem
from rdkit.Chem import AllChem
import random
import numpy as np
import torch
import hashlib
import os
from utils.amino_acid import AminoAcid
import utils.data_constants as ud
from openfold.np.residue_constants import restype_3to1, resname_to_idx
from utils.chem_utils import pre_compute_distance_bound
import shutil
import metis
import networkx as nx
smiles_to_idx={Chem.MolToSmiles(Chem.MolFromSequence(restype_3to1[_], flavor=0)):resname_to_idx[_] for _ in restype_3to1}

def recursive_partition(graph, id=0,depth=0, max_depth=4):
    if depth >= max_depth:
        return [list(graph.nodes), id]
    if len(list(graph.nodes))<2:
        subgraphs=[list(graph.nodes),[]]
    else:
        metis_graph = metis.networkx_to_metis(graph)
    
        edgecuts, partition = metis.part_graph(metis_graph, nparts=2)

        subgraphs = [[], []]
        for node, part in zip(graph.nodes, partition):
            subgraphs[part].append(node)

    subgraph1 = graph.subgraph(subgraphs[0]).copy()
    subgraph2 = graph.subgraph(subgraphs[1]).copy()

    return [recursive_partition(subgraph1,id*2, depth + 1, max_depth),
           recursive_partition(subgraph2,id*2+1, depth + 1, max_depth), id]
    
def get_matrix_for_partions(partions, id_matrixes, depth):
    if len(partions)==2: # leaf node
        nodes, id=partions
        for v in nodes:
            assert id_matrixes[depth][v]<0
            id_matrixes[depth][v]=id
        return
    assert len(partions)==3
    left, right, id=partions
    ori_child_matrix=id_matrixes[depth+1].copy()
    get_matrix_for_partions(left,  id_matrixes, depth+1)
    get_matrix_for_partions(right, id_matrixes, depth+1)
    id_matrixes[depth][id_matrixes[depth+1]!=ori_child_matrix]=id
    
    

def myHash(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def add_part_info(feat, max_level):
    num_node=len(feat["nodes_int_feats"])
    edge=feat["edges"]
    assert edge.shape[-1]==2, edge.shape
    G=nx.Graph()
    G.add_edges_from(edge.tolist())
    partitions=recursive_partition(G, max_depth=max_level)
    part_matrix=[np.zeros(num_node, dtype=int)-1 for _ in range(max_level+1)]
    get_matrix_for_partions(partitions,  part_matrix, 0)
    part_matrix=part_matrix[::-1]
    part_info=[]
    for i,v in enumerate(part_matrix):
        # make v unique
        num_unique_v=len(np.unique(v))
        map_v={a:i for i,a in enumerate(np.unique(v))}
        v=np.array([map_v[_] for _ in v])
        mapped_edge=np.unique(v[edge], axis=0) 
        assert mapped_edge.shape[-1]==2, mapped_edge.shape       
        mapped_edge=mapped_edge[mapped_edge[..., 0]!=mapped_edge[..., 1]]
        remap_v=v.copy()
        if i>0:
            # aggr v
            remap_v=np.zeros(prev_unique_v.max()+1, dtype=int)
            for j in np.unique(prev_unique_v):
                remap_v[j]=v[prev_unique_v==j].max()
        part_info.append([remap_v, mapped_edge])
        prev_unique_v=v
    feat["part_info"]=part_info
    return feat

def get_graph(aa_name=None, smiles=None, max_level=4):
    if smiles is not None:
        
        aa_name=myHash(smiles)
        ligand_path=None
    else:
        
        ligand_path=ud.cache_data_path
    
    
    ret=AminoAcid(aa_name=aa_name, aa_idx=0, smiles=smiles, ligand_path=ligand_path).get_graph_with_gt()
    return ret#add_part_info(ret, max_level)
    


def load_conf(path, smiles, max_level=4):
    name = path.split("/")[-1].replace(".mol", "")
    ret=get_graph(aa_name=name, max_level=max_level)
    if smiles in smiles_to_idx:
        ret["aa_label"]=torch.tensor( smiles_to_idx[smiles]).reshape(-1)
    else:
        ret["aa_label"]=torch.tensor( -1).reshape(-1)
    return ret
    


def get_idx_of_H(atom):
    neighbors = atom.GetNeighbors()
    for v in neighbors:
        if v.GetSymbol() == "H":
            return v.GetIdx()


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


def merge_aa(mid, num_confs, start=None, end=None):
    if start is not None and end is not None:
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
            mol, numConfs=num_confs,  useExpTorsionAnglePrefs=True, useBasicKnowledge=True)

        # build return
        emol = Chem.EditableMol(Chem.Mol())
        ret_map = {}
        for v in mid_map.values():
            ret_map[v] = emol.AddAtom(mol.GetAtomWithIdx(v))
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if start in ret_map and end in ret_map:
                emol.AddBond(
                    ret_map[start], ret_map[end], order=bond.GetBondType())
        ret = []
        rev_ret_map = {v: k for k, v in ret_map.items()}
        for v in cids:
            conf = mol.GetConformer(v)
            tmp = emol.GetMol()
            tmp_conf = AllChem.Conformer(tmp.GetNumAtoms())

            for atom in tmp.GetAtoms():
                if atom.GetIdx() in rev_ret_map:
                    tmp_conf.SetAtomPosition(
                        atom.GetIdx(), conf.GetAtomPosition(rev_ret_map[atom.GetIdx()]))
            tmp.AddConformer(tmp_conf)
            ret.append(tmp)
    else:
        mol = Chem.AddHs(Chem.MolFromSmiles(mid))
        Chem.SanitizeMol(mol)
        cids = AllChem.EmbedMultipleConfs(
            mol, numConfs=num_confs,  useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
        ret = []
        
        for v in cids:
            conf = mol.GetConformer(v)
            tmp =  Chem.AddHs(Chem.MolFromSmiles(mid))
            tmp_conf = AllChem.Conformer(tmp.GetNumAtoms())

            for atom in tmp.GetAtoms():
                tmp_conf.SetAtomPosition(
                        atom.GetIdx(), conf.GetAtomPosition(atom.GetIdx()))
            tmp.AddConformer(tmp_conf)
            ret.append(tmp)
    return ret


class MolDataset(Dataset):
    def __init__(self, aa_path, mol_path=None, cache_path=None, num_projection=20, world_size=1, rank=0, num_level=4) -> None:
        super().__init__()
        aa_data = pd.read_csv(aa_path)
        self.aa_smiles = aa_data["SMILES"]
        self.aa_similarity = aa_data["SIMILARITY"]
        self.aa_neighbors = aa_data["NEIGHBORS"]
        select_pos = ~np.isnan([float(_.split(";")[0])
                               for _ in self.aa_similarity])
        self.aa_smiles = list(self.aa_smiles[select_pos])
        self.aa_similarity = list(self.aa_similarity[select_pos])
        self.aa_neighbors = list(self.aa_neighbors[select_pos])

        self.cache_path = cache_path
        if mol_path is not None:
            self.mol_data = pd.read_csv(mol_path, sep=" ")["SMILES"]
        else:
            self.mol_data = None
        self.num_aa = len(self.aa_smiles)
        self.num_projection = num_projection
        step=(len(self.aa_smiles)+world_size-1)//world_size
        self.index=list(range(len(self.aa_smiles)))[step*rank:step*rank+step]
        self.num_level=num_level
        
        mol_step=(len(self.mol_data)+world_size-1)//world_size
        self.mol_index=list(range(len(self.mol_data)))[step*rank:step*rank+step]
        print(len(self.index), len(self.mol_index))
        
        
    def build_neighbor_key(self):
        map_str_to_idx={}
        for i,v in enumerate(self.aa_smiles):
            map_str_to_idx[v]=i
        ret={}
        for i,v in enumerate(self.aa_neighbors) :
            v=v.split(";")
            ret[i]=set([map_str_to_idx[_] for _ in v if _ in map_str_to_idx])
        return ret
            
        
    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        index=self.index[index]
        mid_aa = self.aa_smiles[index]
        neighbors = self.aa_neighbors[index].split(";")
        if len(neighbors) > 0:
            idx=random.randint(0, len(neighbors)-1)
            neighbor_aa = neighbors[idx]
            sim=torch.tensor(float(self.aa_similarity[index].split(";")[idx])).reshape(-1)
  
        else:
            neighbor_aa = mid_aa
            sim=torch.tensor(1).reshape(-1)
        mol_idx=self.mol_index[index%len(self.mol_index)]
        return get_graph(smiles=mid_aa, max_level=self.num_level), get_graph(self.mol_data[mol_idx], max_level=self.num_level), get_graph(smiles=neighbor_aa, max_level=self.num_level)
       


def collate_fn(batch):
    increase_key = ["edges", "pseudo_backbone_atoms",
                     "pseudo_node_index", "defined_torsion_angle", "local_frame_indicator", "other_edges",]
    increase_node_key=[ "permutation"]
    
    def merge_part_info(part_info):
        num_part_info=len(part_info[0])
        ret=[]
        for i in range(num_part_info):
            info=[_[i] for _ in part_info]
            minfo=[]
            einfo=[]
            binfo=[]
            num_n=0
            for i,v in enumerate(info):
                minfo.append(v[0]+num_n)
                einfo.append(v[1]+num_n)
                assert v[1].shape[-1]==2, v[1].shape
                num_n+=v[0].max()+1
                binfo.append(np.zeros(v[0].max()+1, dtype=int)+i)
        
            ret.append([torch.cat([torch.tensor(_) for _ in minfo], 0), torch.cat([torch.tensor(_) for _ in einfo], 0).transpose(1, 0), torch.cat([torch.tensor(_) for _ in binfo], 0)])
        return ret

    def merge_sub_group(idx):
        group = [_[idx] for _ in batch]
        num_node = 0
        num_raw_node=0
        ret = {"node_residue_index": []}
        #part_info=merge_part_info([_["part_info"] for _ in group])
        #[_.pop("part_info") for _ in group]
        num_residue=0
        for i, g in enumerate(group):
            for k in g:
                if not isinstance(g[k], torch.Tensor):
                    g[k]=torch.tensor(g[k])
                if k not in ret:
                    if k=="edges":
                        if g[k].shape[0]==2:
                            g[k]=g[k].transpose(1, 0)
                    ret[k] = [g[k]]
                elif k in increase_key:
                    if k=="edges":
                        if g[k].shape[0]==2:
                            g[k]=g[k].transpose(1, 0)
                    ret[k].append(g[k]+num_node)
                    ret[k][-1][g[k]<0]=-1
                elif k in increase_node_key:
                    ret[k].append(g[k]+num_raw_node)
                    ret[k][-1][g[k]<0]=-1
                else:
                    if k=="node_residue_index":
                        ret[k].append(g[k]+num_residue)
                    else:
                        ret[k].append(g[k])
            num_node += len(g["nodes_int_feats"])
            num_raw_node += len(g["nodes_int_feats"])-1
            if "node_residue_index" not in g:
                ret["node_residue_index"].append(
                    torch.zeros(len(g["nodes_int_feats"])).long()+i)
            else:
                num_residue=num_residue+g["node_residue_index"].max()+1
            if "local_frame_indicator" in g:
                if "ori_local_frame_indicator" not in ret:
                    ret["ori_local_frame_indicator"]=[]
                ret["ori_local_frame_indicator"].append(g["local_frame_indicator"])
        for k in ret:
            ret[k] = torch.cat(ret[k], 0)
        edge_filter=ret["edges"]
        edge_filter=(edge_filter[..., 0]<num_node)&(edge_filter[..., 1]<num_node)
        ret["edges"] = ret["edges"][edge_filter].transpose(1, 0)
        ret["edge_attrs"]=ret["edge_attrs"][edge_filter]
        #ret["part_info"]=part_info
        return ret
    ret = [merge_sub_group(i) for i in range(len(batch[0]))]
    return ret


def save_pdb(atom_type, atom_position, res_id, save_path):
    assert atom_type.shape == atom_position.shape[:-1]
    assert atom_type.shape == res_id.shape
    for i, idx in enumerate(np.unique(res_id)):
        catom = atom_type[res_id == idx][:-1]
        catom_position = atom_position[res_id == idx][:-1]
        with open(save_path+"_"+str(i)+".pdb", 'w') as f:
            for i, (atom, coord) in enumerate(zip(catom, catom_position), start=1):
                atom = Chem.Atom(int(atom)).GetSymbol()
                f.write(
                    "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format(
                        "ATOM",  # record name
                        i,  # atom serial number
                        atom+f"{i}",  # atom name
                        "",  # alternate location indicator
                        "UNK",  # residue name
                        "A",  # chain identifier
                        1,  # residue sequence number
                        "",  # code for insertion of residues
                        coord[0],  # orthogonal coordinates for X in Angstroms
                        coord[1],  # orthogonal coordinates for Y in Angstroms
                        coord[2],  # orthogonal coordinates for Z in Angstroms
                        1.00,  # occupancy
                        0.00,  # temperature factor
                        atom,  # element symbol
                        ""  # charge on the atom
                    )
                )
            f.write("END\n")
