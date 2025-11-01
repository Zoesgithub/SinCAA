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


pi = torch.tensor(3.141592653589793)


def rev_attr(edge_index):
    # add reverse edges and add attr
    edge_attr = edge_index.new_zeros(edge_index.shape[1])
    edge_index = torch.cat([edge_index, torch.stack([edge_index[1], edge_index[0]], 0)], 1)
    edge_attr = torch.cat([edge_attr, edge_attr + 1], 0).float()[..., None]
    return edge_index, edge_attr


class ResidueGraphModel(nn.Module):
    def __init__(self, args, hidden_size):
        super().__init__()
        self.args = args
        self.aanet_proj = nn.Sequential(
            nn.Linear(512, hidden_size),
        )
        self.convs = nn.ModuleList()
        for _ in range(3):
            l = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                
            )
            self.convs.append(pyg_nn.GINEConv(l))
        self.edge_type_emb=nn.Embedding(100, hidden_size)
        self.out_norm=nn.LayerNorm(hidden_size)

    def forward(self,peptide_feature, edge_index, edge_attr):
        peptide_feat = self.aanet_proj(peptide_feature)
        edge_attr=self.edge_type_emb(edge_attr[..., 0].long())
        assert edge_index.max()==peptide_feat.shape[0]-1
        for l in self.convs:
            peptide_feat=l(peptide_feat, edge_index, edge_attr=edge_attr)+peptide_feat
        return self.out_norm(peptide_feat)

class ProteinGraphModel(nn.Module):
    def __init__(self, args, hidden_size):
        super().__init__()
        self.args = args
        self.chemical_net = nn.Sequential(
            nn.Linear(hidden_size+1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.atom_net_a = nn.Sequential(
            nn.Linear(33, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.atom_net_b = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.surface_mask_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.atomtype_embedding = nn.Embedding(args.vocab_length, 32)
        self.protein_conv = nn.ModuleList()
        self.out_gn=nn.ModuleList()
     
        for _ in range(2):
            l = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                
            )
            self.protein_conv.append(pyg_nn.GINEConv(l))
      
            l = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                
            )
            self.out_gn.append(pyg_nn.GINEConv(l))
            
        self.feat_scale = nn.Linear(hidden_size, hidden_size)
        self.edge_linear=nn.Linear(2, hidden_size)
        self.prot_linear=nn.Linear(1280, hidden_size)
        self.out_norm=nn.LayerNorm(hidden_size)
        self.cross_attr_linear=nn.Linear(1, hidden_size)
        self.pred=nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        
    def forward(self, graph, pro_emb):
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
        edge_attr = self.edge_linear(torch.cat([distance[..., None], graph.edge_attr[..., None]], -1))
        merge_pro_vertex_edge=graph.merge_pro_vertex_edge
        
        n_feat=feat.shape[0]
        n_pro=pro_emb.shape[0]
        #assert merge_pro_vertex_edge[0].max()==pro_emb.shape[0]-1, f"{merge_pro_vertex_edge[0].max()} {pro_emb.shape[0]}"
        assert merge_pro_vertex_edge[1].max()==feat.shape[0]-1, f"{merge_pro_vertex_edge[1].max()} {feat.shape[0]-1}"
        # handle edge
        merge_pro_vertex_edge=torch.cat(
            [graph.pro_edge, torch.stack([merge_pro_vertex_edge[0], merge_pro_vertex_edge[1]+n_pro], 0),torch.stack([ merge_pro_vertex_edge[1]+n_pro, merge_pro_vertex_edge[0]], 0),],
            1
        )
        
        pro_emb= self.prot_linear(pro_emb)
        cross_edge_atrr=self.cross_attr_linear(1/graph.prot_dist)
        in_edge_attr=self.cross_attr_linear(1/graph.prot_ind)
        cross_edge_atrr=torch.cat([in_edge_attr, cross_edge_atrr,cross_edge_atrr],0)
        
        assert  graph.edge_index.max()==feat.shape[0]-1
        feat=self.protein_conv[0](feat, graph.edge_index.long(), edge_attr=edge_attr)+feat
        return self.out_norm(pro_emb), self.pred(feat).squeeze(-1), [feat, merge_pro_vertex_edge, cross_edge_atrr]

class crossmodule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.pro_model=nn.TransformerEncoderLayer(hidden_size, 4, batch_first=True)
        self.pep_model=nn.TransformerEncoderLayer(hidden_size, 4, batch_first=True)
        self.pro_transform=nn.MultiheadAttention(hidden_size, 16, batch_first=True)
        self.pep_transform=nn.MultiheadAttention(hidden_size, 16, batch_first=True)
        self.pair_linear=nn.Linear(32, hidden_size)
        self.pep_linear=nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.pro_linear=nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
       
        l = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            
        )
        self.gn=pyg_nn.GINEConv(l)
        self.pred_surf=nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1), nn.Sigmoid())
    
     
        
    def forward(self, pro_feat, pep_feat, surfinfo, pro_mask, pep_mask):
        n_pro=pro_feat.shape[0]
        feat, merge_pro_vertex_edge, cross_edge_atrr=surfinfo
        
        merge_feat=torch.cat([pro_feat, feat], 0)
        merge_feat=self.gn(merge_feat, merge_pro_vertex_edge.long(), edge_attr=cross_edge_atrr)+merge_feat
        surf_pred=self.pred_surf(merge_feat[n_pro:]).squeeze(-1)
        
        shape=[*pro_mask.shape, pro_feat.shape[-1]]
        batch_pro_feat=pro_feat.new_zeros((shape[0]*shape[1], shape[2]))
        batch_pro_feat[pro_mask.reshape(-1)>0]=merge_feat[:n_pro]
        batch_pro_feat=batch_pro_feat.reshape(shape)

        batch_pro_feat=self.pro_model(batch_pro_feat,  src_key_padding_mask=(1-pro_mask).bool())
        pep_feat=self.pep_model(pep_feat, src_key_padding_mask=(1-pep_mask).bool())
        
        tpep_feat, pep_weight=self.pro_transform(pep_feat, batch_pro_feat, batch_pro_feat,average_attn_weights=False, key_padding_mask=(1-pro_mask).bool())
        tpro_feat, pro_weight=self.pep_transform(batch_pro_feat, pep_feat, pep_feat,average_attn_weights=False, key_padding_mask=(1-pep_mask).bool())
        pair_feat=self.pair_linear(torch.cat([pep_weight, pro_weight.transpose(3,2)], 1).permute(0,2,3,1))
        
        return (batch_pro_feat+self.pro_linear(tpro_feat))[pro_mask>0], pep_feat+self.pep_linear(tpep_feat), pair_feat, surf_pred, [merge_feat[n_pro:], merge_pro_vertex_edge, cross_edge_atrr]
        
        
class SurfaceBind(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size=80
        self.protein_model = ProteinGraphModel(args,self.hidden_size)
        self.peptide_model = ResidueGraphModel(args,self.hidden_size)
        self.args = args
        h=80
        
        self.pro_trans=nn.Linear(self.hidden_size, h)
        self.pep_trans=nn.Linear(self.hidden_size, h)
        self.cross_block=nn.ModuleList([crossmodule(h) for _ in range(3)])
        self.out_layer=nn.Sequential(nn.Linear(h*2, 1), nn.Sigmoid())
        self.pro_pred=nn.Sequential(nn.Linear(h, 1), nn.Sigmoid())
        self.pep_pred=nn.Sequential(nn.Linear(h, 1), nn.Sigmoid())

    def sample_surf(self, surfinfo, npos, pos, ratio=0.1):
        feat, cross_edge, cross_edgeattr=surfinfo
        map_index=torch.zeros(len(feat)).long()-1
        
        select_pos=torch.rand_like(feat[:, 0])<ratio
        map_index[select_pos]=torch.arange(select_pos.long().sum().item())+npos
        feat=feat[select_pos]
        pos=pos[select_pos]
        
        select_pos=torch.cat([select_pos.new_ones(npos), select_pos], 0)
        select_edge=select_pos[cross_edge[0]]&select_pos[cross_edge[1]]
        edge=cross_edge[:, select_edge]
        
        remap=torch.cat([torch.arange(npos), map_index], 0)
        assert edge.max()<remap.shape[0], f'{edge.max()}{remap.shape}'
        edge=remap.to(edge.device)[edge]
        assert edge.min()>-1
       
        return [feat, edge, cross_edgeattr[select_edge]], pos
        
    def forward(self, batch):
        peptide, protein, label, pro_emb = batch
        
        peptide = peptide.to(self.args.device)
        protein = protein.to(self.args.device)
        peptide_feature = self.pep_trans(self.peptide_model(peptide.x, peptide.edge_index, peptide.edge_attr))
        protein_feature, surface_pred, surfinfo =self.protein_model(protein, pro_emb)
        protein_feature= self.pro_trans(protein_feature)
        pep_residue_index=peptide.residue_index
        # cross info
        pep_batch_idx=peptide.batch_info
        pro_batch_idx=protein.batch_info
        
        # cross info
        pred=[]
        pro_pred=[]
        pep_pred=[]
        surface_preds=[]
        pos=protein.x
       
        pro_mask=protein.pro_mask
        pep_mask=peptide.pep_mask
        
        pair_feat=0
        shape=[*pep_mask.shape, peptide_feature.shape[-1]]
        batch_pep_feat=peptide_feature.new_zeros((shape[0]*shape[1], shape[2]))
        batch_pep_feat[pep_mask.reshape(-1)>0]=peptide_feature
        batch_pep_feat=batch_pep_feat.reshape(shape)
        pro_feat=protein_feature
        
        for b in self.cross_block:
            pro_feat, batch_pep_feat, p, sp, surfinfo=b(pro_feat, batch_pep_feat, surfinfo, pro_mask, pep_mask)
            surface_preds.append(sp)
            pair_feat=pair_feat+p
        
        pep_feat=batch_pep_feat[pep_mask>0]
        assert len(pro_feat.shape)==2
        
        for b in range(pep_batch_idx.long().max().item()+1):
            pep_idx=pep_batch_idx==b
            pro_idx=pro_batch_idx==b
            pep_f=pep_feat[pep_idx]
            pro_f=pro_feat[pro_idx]
            pair_f=pair_feat[b][:len(pep_f), :len(pro_f)]
            
            merge_shape=[pep_f.shape[0],pro_f.shape[0],  pro_f.shape[-1]]
            p=self.out_layer(torch.cat([pro_f[None].expand(merge_shape)+pair_f, pep_f[:,None].expand(merge_shape)+pair_f], -1)).squeeze(-1)
            
            pred.append(p)
            pro_pred.append(self.pro_pred(pro_f).squeeze(-1))
            pep_pred.append(self.pep_pred(pep_f).squeeze(-1))
            
        batch=protein.batch
        sloss=0
        for b in range(batch.max().item()+1):
            p=batch==b
            l=torch.cdist(pos[p].float(),peptide.cords[pep_batch_idx==b].float()).min(1).values
            l=(l<4).float()
            for v in surface_preds:
                v=v[p]
                assert l.shape==v.shape
                sloss=sloss+10*(l*torch.log(v.clamp(1e-8))+(1-l)*torch.log((1-v).clamp(1e-8))).mean()
        return pred,pro_pred, pep_pred, -sloss
        