# -*- coding: utf-8 -*-

import json
import os
import time
from argparse import Namespace
from pdb import set_trace
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch, Data

from run.dataset import RESIDUE_DIM, RESIDUE_VOCAB, PeptideGraph
from run.data_utils import collate_fn as smiles_collate
from run.data_utils import get_graph

pi = torch.tensor(3.141592653589793)


def load_model(prefix, device):
    with open(os.path.join(prefix, "config.json"), "r") as f:
        args = Namespace(**json.load(f))
    model = AANet(args)

    cache_state_dict = torch.load(os.path.join(prefix, "model.statedict.best"))
    load_state_dict = {k.replace("module.", ""): cache_state_dict[k] for k in cache_state_dict}
    model.load_state_dict(load_state_dict)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def rev_attr(edge_index):
    # add reverse edges and add attr
    edge_attr = edge_index.new_zeros(edge_index.shape[1])
    edge_index = torch.cat([edge_index, torch.stack([edge_index[1], edge_index[0]], 0)], 1)
    edge_attr = torch.cat([edge_attr, edge_attr + 1], 0).float()[..., None]
    return edge_index, edge_attr


class ResidueGraphModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.residue_embeding = nn.ModuleList([nn.Embedding(n, d) for n, d in zip(RESIDUE_VOCAB, RESIDUE_DIM)])
        self.args = args
        self.aanet_proj = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 80),
        )
      
        self.peptide_conv = pyg_nn.GIN
        Graphormer(
            num_layers=4,
            input_node_dim=512,
            node_dim=512,  # this is hidden channel
            input_edge_dim=1,
            edge_dim=2,
            output_dim=512,
            n_heads=8,
            max_in_degree=5,
            max_out_degree=5,
            max_path_distance=50,
        )
        self.in_net = nn.Linear(80, 512)
        self.out_net = nn.Linear(512, 80)
        self.pe = nn.Embedding(100, 80)

        self.edge_net1 = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8))  # for chemical graph
        self.edge_net2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 8))  # for fully connected graph

    def _smiles_embedding(self, graph: List[Data]):
        node_embeddings = []
        for data in graph:
            smiles = data.smiles
            inp = get_graph(smiles=smiles)
            cpu_feats = smiles_collate([[inp]])[0]
            inp_feats = {k: cpu_feats[k].to(self.args.device) for k in cpu_feats}
            node_emb, pseudo_emb = self.residue_model.calculate_topol_emb(inp_feats, None)
            pseudo_emb = self.aanet_proj(pseudo_emb)
            # cpu_feats["node_emb"]=node_emb.detach().cpu()
            # cpu_feats["pseudo_emb"]=pseudo_emb
            node_embeddings.append(pseudo_emb)
        return torch.cat(node_embeddings, dim=0)

    def forward(self, graph_list, peptide_edge_index, residue_split_id, peptide_feature):
        """embedding the residue graph using GIN
        # residue graph
        residue_graphs = Batch.from_data_list(graph_list)
        input_x = residue_graphs.x
        num_residues = residue_graphs.batch.max() + 1
        num_node = input_x.shape[0]
        input_x = torch.cat([input_x, input_x.new_zeros((num_residues, input_x.shape[1]))], 0)
        atom_embeds = [self.residue_embeding[i](input_x[:, i]) for i in range(len(RESIDUE_VOCAB))]

        # concat 2 kinds of attributes and index
        edge_attr = torch.cat(
            [self.edge_net1(residue_graphs.edge_attr), self.edge_net2(residue_graphs.edge_attr_fc)], 0
        )
        edges = torch.cat([residue_graphs.edge_index, residue_graphs.edge_index_fc], -1).long()
        # todo to learnable edge feature

        new_edges = torch.stack([torch.arange(num_node).to(input_x.device), residue_graphs.batch + num_node], 0)
        rev_new_edges = torch.stack([residue_graphs.batch + num_node, torch.arange(num_node).to(input_x.device)], 0)
        new_edges = torch.cat([new_edges, rev_new_edges], 1)

        edges = torch.cat([edges, new_edges], 1)
        edge_attr = torch.cat([edge_attr, edge_attr.new_zeros([new_edges.shape[0], edge_attr.shape[1]])])
        residue_feat = torch.cat(atom_embeds, dim=1)
        residue_feat = self.residue_conv(residue_feat, edges, edge_attr=edge_attr)

        peptide_feat = residue_feat[num_node:]  # context-free feature
        """
        peptide_feat = self.aanet_proj(peptide_feature)
        # construct another batch for graphormer
        peptide_feat_split = torch.split(peptide_feat, residue_split_id, dim=0)
        peptide_feature = []

        for i in range(len(peptide_feat_split)):
            in_feat = self.in_net(peptide_feat_split[i])
            in_graph = Data(x=in_feat, edge_index=peptide_edge_index[i])
            in_feat = self.peptide_conv.forward(in_graph)
            peptide_feature.append(self.out_net(in_feat))
        peptide_mask = torch.cat(peptide_feature, 0)  # context mask

        assert peptide_mask.shape == peptide_feat.shape
        return peptide_mask, peptide_feat


class ProteinGraphModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.chemical_net = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 15),
        )

        self.atom_net_a = nn.Sequential(
            nn.Linear(33, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 15),
        )

        self.atom_net_b = nn.Sequential(
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 15),
        )

        self.surface_mask_net = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 80),
        )

        self.atomtype_embedding = nn.Embedding(args.vocab_length, 32)
        self.protein_conv = nn.ModuleList([pyg_nn.GIN(80, 256, 2, 80, norm="LayerNorm") for _ in range(5)])
        self.edge_attr_to_atom_feats = nn.ModuleList(
            [nn.Sequential(nn.Linear(82, 80), nn.ReLU(), nn.Linear(80, 80)) for _ in range(5)]
        )

        self.feat_scale = nn.Linear(15, 80)

    def forward(self, graph):
        pos = torch.tensor(np.concatenate([p for p in graph.mesh_vertex], 0)).float().to(self.args.device)
        graph = graph.to(self.args.device)
        atom_cat = (
            torch.from_numpy(np.concatenate([x for x in graph.atom_feat], 0)).float().to(self.args.device).squeeze(-1)
        )
        dist_cat = torch.from_numpy(np.concatenate([x for x in graph.atom_dist], 0)).float().to(self.args.device)

        atom_feat = self.atomtype_embedding(atom_cat.long())
        atom_dist = dist_cat[..., None]

        data = graph.x
        geom_feature = data[:, 0][..., None].clone()

        feat_all = torch.cat([atom_feat, 1 / atom_dist], dim=-1)

        atom_dist_feat = self.atom_net_a(feat_all)
        atom_dist_feat = atom_dist_feat.sum(1)
        atom_dist_feat = self.atom_net_b(atom_dist_feat)
        feat = torch.cat([atom_dist_feat, geom_feature], -1)
        feat = self.chemical_net(feat)
        feat = self.feat_scale(feat)

        # aggregate_edge_feats
        spos = pos[graph.edge_index.long()[0]]
        epos = pos[graph.edge_index.long()[1]]
        distance = ((spos - epos) ** 2).sum(-1).sqrt()
        edge_attr = torch.cat([distance[..., None], graph.edge_attr[..., None]], -1)
        for m, ea in zip(self.protein_conv, self.edge_attr_to_atom_feats):
            edge_feat = ea(torch.cat([edge_attr, feat[graph.edge_index.long()[1]]], 1))
            atom_edge_feat = torch.scatter_reduce(
                torch.zeros_like(feat),
                0,
                graph.edge_index.long()[0][..., None].expand_as(edge_feat),
                edge_feat,
                reduce="sum",
            )

            feat = m(feat + atom_edge_feat, graph.edge_index.long())
        surface_mask = self.surface_mask_net(feat)

        return surface_mask, feat


class SurfaceBind(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.peptide_model = ResidueGraphModel(args)
        self.protein_model = ProteinGraphModel(args)
        self.args = args
        self.peptide_sigmoid = nn.Sequential(nn.Linear(80, 128), nn.ReLU(), nn.Linear(128, 80), nn.Sigmoid())
        self.protein_sigmoid = nn.Sequential(nn.Linear(80, 128), nn.ReLU(), nn.Linear(128, 80), nn.Sigmoid())
        self.pro_linear = nn.Sequential(
            nn.Linear(80 * 2, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        peptide, peptide_index, protein, residue_split_id, label, peptide_pretrained = batch

        peptide_index = [idx.to(torch.int32).to(self.args.device) for idx in peptide_index]
        peptide = [pp.to(self.args.device) for pp in peptide]
        protein = protein.to(self.args.device)
        peptide_pretrained = peptide_pretrained.to(self.args.device)

        context_mask, peptide_feature = self.peptide_model(peptide, peptide_index, residue_split_id, peptide_pretrained)
        context_mask_feature = peptide_feature * context_mask

        surface_mask, protein_feature = self.protein_model(protein)
        pos, neg = label

        peptide_pos_idx = np.array([x[0] for x in pos])
        protein_pos_idx = np.array([x[1] for x in pos])
        peptide_neg_idx = np.array([x[0] for x in neg])
        protein_neg_idx = np.array([x[1] for x in neg])

        peptide_pos = context_mask_feature[peptide_pos_idx]
        protein_pos = protein_feature[protein_pos_idx]
        surface_mask_pos = surface_mask[protein_pos_idx]
        masked_pep_pos = peptide_pos * surface_mask_pos

        peptide_neg = context_mask_feature[peptide_neg_idx]
        protein_neg = protein_feature[protein_neg_idx]
        surface_mask_neg = surface_mask[protein_neg_idx]
        masked_pep_neg = peptide_neg * surface_mask_neg

        pos_pred = self.pro_linear(torch.cat([masked_pep_pos, protein_pos], 1))
        neg_pred = self.pro_linear(torch.cat([masked_pep_neg, protein_neg], 1))

        return (pos_pred, neg_pred, masked_pep_pos, masked_pep_neg)
