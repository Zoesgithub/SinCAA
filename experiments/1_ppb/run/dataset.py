# -*- coding: utf-8 -*-
import os.path as osp
import os
from typing import List

import Bio.PDB as biopdb
import numpy as np
import pytorch_lightning as pl
import torch
from scipy.spatial import KDTree
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from protein.chemical_feature import ele2num
from utils.prepare_data import PeptideGraph
from peptide.graph import SPeptideChain, HPeptideChain
import time
from scipy.spatial.distance import cdist
import pickle
import json
import pandas as pd
# args = parser.parse_args()
RESIDUE_VOCAB = [13, 3, 5, 7, 4, 2, 5, 2, 2]
RESIDUE_DIM = [4] * len(RESIDUE_VOCAB)  # todo modify this in experiment


def collate_fn(batch: list, data_type=None):
    """
    Ensemble protein and peptide data into a batch
    """
    peptide_graph: List[PeptideGraph] = [x[0] for x in batch]
    protein_graph: List[Data] = [x[1] for x in batch]
    # map_pro_to_residue=[x[2] for x in batch]
    label = [x[2] for x in batch]
    pro_emb = [x[3] for x in batch]
    pro_chaininfo = [x[4] for x in batch]

    peptide_graph_list = []
    num_residues = 0
    merged_residue_index = []
    merged_pep_batch_index = []
    merged_pep_label = []
    merged_label_residue_index = []
    for i, d in enumerate(peptide_graph):
        peptide_graph_list.append(
            Data(x=torch.from_numpy(d.embs).float(),  edge_index=torch.from_numpy(
                d.edge_index).long().transpose(1, 0), edge_attr=torch.from_numpy(d.edge_attr).long())
        )
        merged_residue_index.append(torch.from_numpy(d.residue_index).long())
        merged_pep_batch_index.append(torch.zeros(d.embs.shape[0])+i)
        merged_pep_label.append(torch.tensor(label[i].max(1)).float())
        num_residues = num_residues+d.residue_index.max()+1
        if hasattr(d, "label_residue_index"):
            merged_label_residue_index.append(
                torch.from_numpy(d.label_residue_index))
    pep_mask = np.zeros((len(merged_pep_batch_index), max(
        [len(_) for _ in merged_pep_batch_index])))
    for i, v in enumerate(merged_pep_batch_index):
        pep_mask[i, :len(v)] = 1

    merged_pro_label = []
    merged_pro_batch_index = []
    num_residues = 0
    # num_vertex=0
    # merge_pro_vertex_edge=[]
    # merge_pro_edge=[]
    # merge_new_vertices=[]
    for i, p in enumerate(protein_graph):

        pc = pro_chaininfo[i]
        mapd = {}
        pc_ind = []
        for v in pc:
            if v not in mapd:
                mapd[v] = len(mapd)
            pc_ind.append(mapd[v])
        p.edge_feat = torch.from_numpy(p.edge_feat).float()
        p.node_feat = torch.from_numpy(p.node_feat).float()
        p.neighbor_indices = torch.from_numpy(p.neighbor_indices).long()
        p.pro_chaininfo = torch.from_numpy(np.array(pc_ind)).long()
        merged_pro_label.append(torch.tensor(label[i].max(0)).float())
        merged_pro_batch_index.append(torch.zeros(label[i].shape[1])+i)

    pro_mask = np.zeros((len(merged_pro_batch_index), max(
        [len(_) for _ in merged_pro_batch_index])))
    for i, v in enumerate(merged_pro_batch_index):
        pro_mask[i, :len(v)] = 1

    protein_graph_batch = Batch.from_data_list(protein_graph)
    protein_graph_batch.batch_info = torch.cat(merged_pro_batch_index, 0)
    # protein_graph_batch.merge_pro_vertex_edge=torch.cat(merge_pro_vertex_edge, 1)
    protein_graph_batch.binary_label = torch.cat(merged_pro_label, 0)

    peptide_graph_batch = Batch.from_data_list(peptide_graph_list)
    peptide_graph_batch.residue_index = merged_residue_index
    peptide_graph_batch.batch_info = torch.cat(merged_pep_batch_index, 0)
    peptide_graph_batch.binary_label = torch.cat(merged_pep_label, 0)
    if len(merged_label_residue_index) > 0:
        peptide_graph_batch.label_residue_index = merged_label_residue_index

    return peptide_graph_batch, protein_graph_batch, label, torch.cat([torch.from_numpy(_) for _ in pro_emb], 0)


class MultiLevelGraph(Dataset):
    def __init__(self, data_type, data_dir, processed_dir, threshold=None, inference_file=None, seed=None, extra_emb=False):
        self.data_dir = data_dir
        if not inference_file:
            self.list_path = osp.join(processed_dir, f"{data_type}.txt")
        else:
            self.list_path = inference_file

        self.data_type = data_type
        self.seed = seed
        self.data_list = [x.strip() for x in open(self.list_path)]
        with open("../../../SinCAA_baseline/AANet_label/bs_inconsistent_sample.json", "r") as f:
            filter_files = json.load(f)
        print(len(self.data_list))
        self.data_list = [_ for _ in self.data_list if _ not in filter_files]
        print(len(self.data_list))
        self.length = len(self.data_list)
        self.threshold = threshold
        self.num_none = 0
        self.extra_emb = extra_emb
        self.pep_endfix = ""
        print("peptide_endfix", self.pep_endfix)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        prefix = osp.join(self.data_dir, data, data)

        # protein_surface = torch.load(prefix + "_protein_graph.pt")
        stime = time.time()
        protein_info = torch.load(
            prefix + "_protein_info.pt", weights_only=False)
        peptide_graph = None
        if os.path.exists(prefix+f"_peptide_info_simplify{self.pep_endfix}.pt"):
            try:
                peptide_graph = torch.load(
                    prefix+f"_peptide_info_simplify{self.pep_endfix}.pt", weights_only=False)
            except:
                peptide_graph = None
        if peptide_graph is None:
            peptide_graph = torch.load(
                prefix + f"_peptide_info{self.pep_endfix}.pt", weights_only=False)
            peptide_graph = SPeptideChain(embs=peptide_graph.embs, edge_index=peptide_graph.edge_index, edge_attr=peptide_graph.edge_attr,
                                          atom_types=peptide_graph.atom_types, residue_index=peptide_graph.residue_index, cords=peptide_graph.cords)
            torch.save(peptide_graph, prefix +
                       f"_peptide_info_simplify{self.pep_endfix}.pt")
        protein_emb = torch.load(
            osp.join(self.data_dir, data, "protein_esm_emb.npy"), weights_only=False)
        if self.extra_emb:
            pdb_id = f'{data.split("_")[0].lower()}_{data.split("_")[1]}'

            path = os.path.join("data/peptide_data/higmae_emb/", pdb_id)
            with open(path, "rb") as f:
                pep_emb = pickle.load(f)
            assert np.array(pep_emb["nodes_residue_index"]).max(
            ) == peptide_graph.residue_index.max()
            peptide_graph = HPeptideChain(embs=pep_emb["node_emb"], edge_index=np.array(pep_emb["edges"]), edge_attr=np.array(
                pep_emb["edge_attrs"]), atom_types=peptide_graph.atom_types, residue_index=np.array(pep_emb["nodes_residue_index"]), cords=peptide_graph.cords, label_residue_index=peptide_graph.residue_index)
        # protein_surface_nearest_residue=protein_surface.nearest_residue

        # handling label files for different threshold
        residue_label = torch.load(
            prefix+f"_residue_label.pt", weights_only=False)

        residue_label_x = residue_label.x

        residue_label_x_m = residue_label_x < 0
        residue_label_x = (residue_label_x <= 3).astype(float)
        residue_label_x[residue_label_x_m] = -1
        protein_chain_id = [_[0] for _ in residue_label.row_info]
        # protein_info.protein_chain_id=protein_chain_id
        return peptide_graph, protein_info, residue_label_x, protein_emb, protein_chain_id

    def __len__(self):
        return self.length


class SurfBindDataModule(pl.LightningDataModule):
    """Dataset used for pl training, a tidy way in Main."""

    def __init__(
        self,
        data_dir,
        processed_dir,
        exper_setting,
        fold,
        batch_size,
        num_workers,
        seed,
        threshold,
        extra_emb=False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.processed_dir = osp.join(processed_dir, exper_setting, fold)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.trainset = MultiLevelGraph(
            "train", self.data_dir, self.processed_dir, threshold=threshold, extra_emb=extra_emb)
        self.valset = MultiLevelGraph(
            "val", self.data_dir, self.processed_dir, seed=self.seed, threshold=threshold, extra_emb=extra_emb)
        self.testset = MultiLevelGraph(
            "test", self.data_dir, self.processed_dir, seed=self.seed, threshold=threshold, extra_emb=extra_emb)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            collate_fn=lambda batch: collate_fn(batch, "train"),
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )


class ChugaiGraph(Dataset):
    def __init__(self, cache_path):
        self.pro_file = "7YV1_I_A"
        # read peptide file
        self.pd_file = pd.read_csv(
            "/ibex/user/xuc0d/data/code/SinCAA/experiments/0_benchmark/chugai/230620_2cluster.csv")
        self.peptide_feat = torch.load(cache_path)
        # self.ref_pep_feat=torch.load("/ibex/user/xuc0d/data/code/SinCAA/experiments/send_to_ibex/withres.pt")
        self.data_dir = "data/peptide_data/processed_cv_pepnn/"
        self.length = len(self.peptide_feat["node_embs"])

    def __getitem__(self, idx):
        prefix = osp.join(self.data_dir, self.pro_file, self.pro_file)

        protein_info = torch.load(
            prefix + "_protein_info.pt", weights_only=False)
        peptide_graph = SPeptideChain(embs=self.peptide_feat["node_embs"][idx].detach().cpu().numpy(), edge_index=self.peptide_feat["edges"][idx].detach().cpu().numpy(
        ), edge_attr=self.peptide_feat["edge_attrs"][idx].detach().cpu().numpy(), atom_types=None, residue_index=np.arange(len(self.peptide_feat["node_embs"][idx])), cords=None)

        protein_emb = torch.load(osp.join(
            self.data_dir, self.pro_file, "protein_esm_emb.npy"), weights_only=False)

        residue_label = torch.load(
            prefix+f"_residue_label.pt", weights_only=False)

        residue_label_x = residue_label.x

        residue_label_x_m = residue_label_x < 0
        residue_label_x = (residue_label_x <= 3).astype(float)
        residue_label_x[residue_label_x_m] = -1
        protein_chain_id = [_[0] for _ in residue_label.row_info]
        label = -np.log(self.pd_file["KRAS_KD(M)"][idx])
        return peptide_graph, protein_info, residue_label_x*0+label, protein_emb, protein_chain_id

    def __len__(self):
        return self.length


class ChugaiDataModule(pl.LightningDataModule):
    """Dataset used for pl training, a tidy way in Main."""

    def __init__(
        self,
        cache_path,
        num_workers=0,
        batch_size=1
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.trainset = ChugaiGraph(cache_path=cache_path)
        self.valset = ChugaiGraph(cache_path=cache_path)
        self.testset = ChugaiGraph(cache_path=cache_path)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            collate_fn=lambda batch: collate_fn(batch, "train"),
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )
