from torch.utils.data import Dataset
import pandas as pd
from rdkit import Chem
import random
import numpy as np
import torch
import hashlib
from utils.amino_acid import AminoAcid
import utils.data_constants as ud
from openfold.np.residue_constants import restype_3to1, resname_to_idx

smiles_to_idx = {Chem.MolToSmiles(Chem.MolFromSequence(
    restype_3to1[_], flavor=0)): resname_to_idx[_] for _ in restype_3to1}


def myHash(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_graph(aa_name=None, smiles=None, max_level=4):
    if smiles is not None:
        aa_name = myHash(smiles)
        ligand_path = None
    else:
        ligand_path = ud.cache_data_path

    ret = AminoAcid(aa_name=aa_name, aa_idx=0, smiles=smiles,
                    ligand_path=ligand_path).get_graph_with_gt()
    return ret


def load_conf(path, smiles, max_level=4):
    name = path.split("/")[-1].replace(".mol", "")
    ret = get_graph(aa_name=name, max_level=max_level)
    if smiles in smiles_to_idx:
        ret["aa_label"] = torch.tensor(smiles_to_idx[smiles]).reshape(-1)
    else:
        ret["aa_label"] = torch.tensor(-1).reshape(-1)
    return ret


class MolDataset(Dataset):
    def __init__(self, aa_path, mol_path=None, cache_path=None, num_projection=20, world_size=1, rank=0, max_combine=4, istrain=False) -> None:
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
            self.mol_data = pd.read_csv(mol_path)["SMILES"]
        else:
            self.mol_data = None
        self.num_aa = len(self.aa_smiles)
        self.num_projection = num_projection
        step = (len(self.aa_smiles)+world_size-1)//world_size
        self.index = list(range(len(self.aa_smiles)))[step*rank:step*rank+step]
        self.max_combine = max_combine

        mol_step = (len(self.mol_data)+world_size-1)//world_size
        self.mol_index = list(range(len(self.mol_data)))[
            mol_step*rank:mol_step*rank+mol_step]
        self.istrain = istrain

    def build_neighbor_key(self):
        map_str_to_idx = {}
        for i, v in enumerate(self.aa_smiles):
            map_str_to_idx[v] = i
        ret = {}
        for i, v in enumerate(self.aa_neighbors):
            v = v.split(";")
            ret[i] = set([map_str_to_idx[_] for _ in v if _ in map_str_to_idx])
        return ret

    def __len__(self):
        if not self.istrain:
            return len(self.index)
        return max(len(self.index), len(self.mol_index))

    def __getitem__(self, index):
        mol_idx = self.mol_index[index % len(self.mol_index)]
        index = self.index[index % len(self.index)]
        mid_aa = self.aa_smiles[index]
        neighbors = self.aa_neighbors[index].split(";")
        assert len(neighbors) > 0
        idx = random.randint(0, len(neighbors)-1)
        neighbor_aa = neighbors[idx]
        sim = torch.tensor(
            float(self.aa_similarity[index].split(";")[idx])).reshape(-1)
        if sim == 0:
            print(self.aa_similarity[index], mid_aa)

        aa_data = get_graph(smiles=mid_aa, max_level=None)
        mol_data = get_graph(smiles=self.mol_data[mol_idx], max_level=None)
        nei_data = get_graph(smiles=neighbor_aa, max_level=None)
        aa_data["sim"] = sim
        aa_data["index"] = torch.tensor(index).reshape(-1)
        return aa_data, mol_data, nei_data


class ChainDataset(MolDataset):
    def __getitem__(self, index):
        mol_idx = self.mol_index[index % len(self.mol_index)]
        index = self.index[index % len(self.index)]
        mid_aa = self.aa_smiles[index]
        neighbors = self.aa_neighbors[index].split(";")
        assert len(neighbors) > 0
        idx = random.randint(0, len(neighbors)-1)
        neighbor_aa = neighbors[idx]
        sim = float(self.aa_similarity[index].split(";")[idx])
        # if sim==0:
        #    print(self.aa_similarity[index], mid_aa)

        def remove_last_atom(data):
            ret = {
                "nodes_int_feats": data["nodes_int_feats"][:-1],
                "nodes_float_feats": data["nodes_float_feats"][:-1],
                "edge_attrs": data["edge_attrs"],
                "edges": data["edges"],
            }
            return ret

        sims = [sim]
        indexs = [index]
        aa_data = [get_graph(smiles=mid_aa, max_level=None)]
        nei_data = [get_graph(smiles=neighbor_aa, max_level=None)]
        neg_data = [get_graph(smiles=neighbor_aa, max_level=None)]
        num_added_aa = self.max_combine  # random.randint(0, self.max_combine)
        while num_added_aa > 0:
            num_added_aa -= 1
            aa_data[-1] = remove_last_atom(aa_data[-1])
            nei_data[-1] = remove_last_atom(nei_data[-1])
            neg_data[-1] = remove_last_atom(neg_data[-1])
            idx = self.index[random.randint(0, len(self.index)-1)]
            neighbors = self.aa_neighbors[idx].split(";")
            nidx = random.randint(0, len(neighbors)-1)
            neg = self.index[random.randint(0, len(self.index)-1)]
            aa_data.append(
                get_graph(smiles=self.aa_smiles[idx], max_level=None))
            nei_data.append(get_graph(smiles=neighbors[nidx], max_level=None))
            neg_data.append(
                get_graph(smiles=self.aa_smiles[neg], max_level=None))
            sims.append(float(self.aa_similarity[idx].split(";")[nidx]))
            indexs.append(idx)

        mol_data = get_graph(smiles=self.mol_data[mol_idx], max_level=None)

        aa_data, nei_data, neg_data = collate_fn(
            [[a, b, c] for a, b, c in zip(aa_data, nei_data, neg_data)])

        aa_data["sim"] = torch.tensor(sims).reshape(-1)
        aa_data["batch_sim"] = torch.tensor(min(sims)).reshape(-1)
        aa_data["index"] = torch.tensor(indexs).reshape(-1)

        aa_data["batch_id"] = np.zeros(
            len(aa_data["nodes_int_feats"]), dtype=int)
        nei_data["batch_id"] = np.zeros(
            len(nei_data["nodes_int_feats"]), dtype=int)
        neg_data["batch_id"] = np.zeros(
            len(neg_data["nodes_int_feats"]), dtype=int)
        mol_data["batch_id"] = np.zeros(
            len(mol_data["nodes_int_feats"]), dtype=int)
        return aa_data, mol_data, nei_data, neg_data


def collate_fn(batch):
    increase_key = ["edges", "pseudo_backbone_atoms",
                    "pseudo_node_index", "defined_torsion_angle", "local_frame_indicator", "other_edges",]
    increase_node_key = ["batch_id"]

    def merge_part_info(part_info):
        num_part_info = len(part_info[0])
        ret = []
        for i in range(num_part_info):
            info = [_[i] for _ in part_info]
            minfo = []

            num_n = 0
            for i, v in enumerate(info):
                minfo.append(v[0]+num_n)

                num_n += v[0].max()+1

            ret.append([torch.cat([torch.tensor(_) for _ in minfo], 0)])
        return ret

    def merge_sub_group(idx):
        group = [_[idx] for _ in batch]
        num_node = 0
        ret = {"node_residue_index": []}
        if "part_info" in group[0]:
            part_info = merge_part_info([_["part_info"] for _ in group])
            [_.pop("part_info") for _ in group]
        else:
            part_info = None
        num_residue = 0
        for i, g in enumerate(group):
            for k in g:
                if not isinstance(g[k], torch.Tensor):
                    g[k] = torch.tensor(g[k])
                if k not in ret:
                    if k == "edges":
                        if g[k].shape[0] == 2:
                            g[k] = g[k].transpose(1, 0)
                    ret[k] = [g[k]]
                elif k in increase_key:
                    if k == "edges":
                        if g[k].shape[0] == 2:
                            g[k] = g[k].transpose(1, 0)
                    ret[k].append(g[k]+num_node)
                    ret[k][-1][g[k] < 0] = -1
                elif k in increase_node_key:
                    num_raw_node = ret[k][-1].max()+1
                    ret[k].append(g[k]+num_raw_node)
                else:
                    if k == "node_residue_index":
                        ret[k].append(g[k]+num_residue)
                    else:
                        ret[k].append(g[k])
            num_node += len(g["nodes_int_feats"])

            if "node_residue_index" not in g:
                ret["node_residue_index"].append(
                    torch.zeros(len(g["nodes_int_feats"])).long()+num_residue)
            num_residue = ret["node_residue_index"][-1].max()+1

        for k in ret:
            ret[k] = torch.cat(ret[k], 0)

        ret["edges"] = ret["edges"].transpose(1, 0)

        if part_info is not None:
            ret["part_info"] = part_info
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
