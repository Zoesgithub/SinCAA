# -*- coding: utf-8 -*-
"""
@File   :  data_prepare.py
@Time   :  2024/02/06 18:31
@Author :  Yufan Liu
@Desc   :  Prepare and cache input data
"""


import os.path as osp
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

# args = parser.parse_args()
RESIDUE_VOCAB = [13, 3, 5, 7, 4, 2, 5, 2, 2]
RESIDUE_DIM = [4] * len(RESIDUE_VOCAB)  # todo modify this in experiment


def collate_fn(batch: list, data_type=None):
    """
    Ensemble protein and peptide data into a batch
    """
    peptide_graph: List[PeptideGraph] = [x[0] for x in batch]
    protein_graph: List[Data] = [x[1] for x in batch]
    label = [x[2] for x in batch]
    peptide_feature: List[torch.Tensor] = [x[3] for x in batch]
    assert len(peptide_graph) == len(label)

    # map label
    total_pos = []
    total_neg = []
    m1, m2 = 0, 0
    for idx, pair in enumerate(label):
        pos, neg = pair
        for p in pos:
            total_pos.append((p[0] + m1, p[1] + m2))
        for n in neg:
            total_neg.append((n[0] + m1, n[1] + m2))

        m1 += len(peptide_graph[idx].residue_graph)
        m2 += len(protein_graph[idx].x)

    # shuffle negative pairs avoiding bias
    # this introduce non-paried data in a batch
    # would let model only see negative pairs in a bound complex
    if data_type == "train":
        neg_ligands = [n[0] for n in total_neg]
        neg_targets = [n[1] for n in total_neg]
        np.random.shuffle(neg_targets)
        total_neg = tuple(zip(neg_ligands, neg_targets))
    label = (total_pos, total_neg)

    residue_graphs = [i for x in peptide_graph for i in x.residue_graph]
    residue_split_id = [len(x.residue_graph) for x in peptide_graph]
    peptide_indices = [peptide_graph[i].peptide_edge_index for i in range(len(peptide_graph))]
    peptide_feature = torch.cat(peptide_feature, dim=0)

    protein_graph_batch = Batch.from_data_list(protein_graph)

    return residue_graphs, peptide_indices, protein_graph_batch, residue_split_id, label, peptide_feature


class MultiLevelGraph(Dataset):
    def __init__(self, data_type, data_dir, processed_dir, inference_file=None, seed=None):
        self.data_dir = data_dir
        if not inference_file:
            self.list_path = osp.join(processed_dir, f"{data_type}.txt")
        else:
            self.list_path = inference_file
        self.data_type = data_type
        self.seed = seed
        self.data_list = [x.strip() for x in open(self.list_path)]
        self.length = len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        prefix = osp.join(self.data_dir, data, data)
        peptide_graph = torch.load(prefix + "_peptide_graph.pt")
        peptide_feature = torch.load(prefix + "_aanet_pseudo_feature.pt")  # pretrained feature of peptide
        protein_surface = torch.load(prefix + "_protein_graph_new.pt")
        label = torch.load(prefix + "_label.pt")
        pos, neg = label

        num_to_select = min(len(pos), len(neg))

        if self.seed:
            np.random.seed(self.seed)

        neg_sele = np.random.choice(
            range(len(neg)), size=num_to_select, replace=False
        )  # do this selection in every loading time
        neg = [neg[i] for i in neg_sele]
        label = (pos, neg)

        return peptide_graph, protein_surface, label, peptide_feature

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
    ):
        super().__init__()
        self.data_dir = data_dir
        self.processed_dir = osp.join(processed_dir, exper_setting, fold)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.trainset = MultiLevelGraph("train", self.data_dir, self.processed_dir)
        self.valset = MultiLevelGraph("val", self.data_dir, self.processed_dir, self.seed)
        self.testset = MultiLevelGraph("test", self.data_dir, self.processed_dir, self.seed)

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
