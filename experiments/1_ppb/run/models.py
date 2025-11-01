import numpy as np
import torch
import torch.nn as nn
from .layers import *
from .modules import *
from protein.pepnn_compute_features import node_features, edge_features
import torch_geometric.nn as pyg_nn


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def get_neighbor_features(nodes, neighbor_idx):
    '''
    function from Ingraham et al. 2019 source code

    '''

    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(
        list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def cat_neighbors_nodes(nodes, neighbors, neighbor_indices):

    nodes = get_neighbor_features(nodes, neighbor_indices)
    edge_features = torch.cat([neighbors, nodes], -1)

    return edge_features


class RepeatedModule(nn.Module):

    def __init__(self, edge_features, node_features, n_layers, d_model,
                 n_head, d_k, d_v, d_inner, dropout=0.1):

        super().__init__()

        self.edge_embedding = nn.Linear(edge_features, d_model)
        self.bert_embedding = nn.Linear(1280, node_features)
        self.node_embedding = nn.Linear(node_features*2, d_model)
        # self.sequence_embedding = nn.Embedding(20, d_model)
        self.d_model = d_model

        self.reciprocal_layer_stack = nn.ModuleList([
            ReciprocalLayer(d_model,  d_inner,  n_head,
                            d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def _positional_embedding(self, batches, number):

        result = torch.exp(torch.arange(
            0, self.d_model, 2, dtype=torch.float32)*-1*(np.log(10000)/self.d_model))

        numbers = torch.arange(0, number, dtype=torch.float32)

        numbers = numbers.unsqueeze(0)

        numbers = numbers.unsqueeze(2)

        result = numbers*result

        result = torch.cat((torch.sin(result), torch.cos(result)), 2)

        return result

    def forward(self, peptide_sequence, nodes, edges, neighbor_indices, bert_features, pro_chaininfo):

        bert_features = self.bert_embedding(bert_features)

        sequence_attention_list = []

        graph_attention_list = []

        graph_seq_attention_list = []

        seq_graph_attention_list = []

        sequence_enc = peptide_sequence

        # sequence_enc = self.dropout(sequence_enc)

        encoded_edges = self.edge_embedding(edges)
        encoded_nodes = self.node_embedding(
            torch.cat([nodes, bert_features], dim=-1))

        node_enc = encoded_nodes

        edge_input = cat_neighbors_nodes(encoded_nodes, encoded_edges,
                                         neighbor_indices)

        node_enc = self.dropout_2(node_enc)

        edge_input = self.dropout_3(edge_input)

        for reciprocal_layer in self.reciprocal_layer_stack:

            node_enc, sequence_enc, graph_attention, sequence_attention, node_seq_attention, seq_node_attention =\
                reciprocal_layer(sequence_enc, node_enc,
                                 edge_input, pro_chaininfo)

            sequence_attention_list.append(sequence_attention)

            graph_attention_list.append(graph_attention)

            graph_seq_attention_list.append(node_seq_attention)

            seq_graph_attention_list.append(seq_node_attention)

            edge_input = cat_neighbors_nodes(node_enc, encoded_edges,
                                             neighbor_indices)

        return node_enc, sequence_enc, sequence_attention_list, graph_attention_list, \
            seq_graph_attention_list, graph_seq_attention_list


class ResidueGraphModel(nn.Module):
    def __init__(self, args, hidden_size):
        super().__init__()
        self.args = args
        if args.extra_emb:
            self.aanet_proj = nn.Sequential(
                nn.Linear(300, hidden_size),
            )
        else:
            self.aanet_proj = nn.Sequential(
                nn.Linear(512, hidden_size),
            )
        self.convs = nn.ModuleList()
        for _ in range(2):
            l = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            self.convs.append(pyg_nn.GINEConv(l))
        self.edge_type_emb = nn.Embedding(100, hidden_size)
        self.cls_token = nn.Embedding(1, hidden_size)
        self.cls_edge = nn.Embedding(1, hidden_size)

    def forward(self, peptide_feature, edge_index, edge_attr):
        peptide_feat = self.aanet_proj(peptide_feature)
        if len(edge_attr.shape) == 2:
            edge_attr = self.edge_type_emb(edge_attr[..., 0].long())
        else:
            edge_attr = self.edge_type_emb(edge_attr.long())
        # assert edge_index.max()==peptide_feat.shape[0]-1
        # add a node
        peptide_feat = torch.cat([peptide_feat, self.cls_token.weight], 0)
        edge_index = torch.cat([
            edge_index, torch.stack([torch.arange(len(
                peptide_feat)-1).to(peptide_feat.device), peptide_feat.new_zeros(len(peptide_feat)-1).long()+len(peptide_feat)-1], 0)
        ], 1)

        edge_attr = torch.cat(
            [edge_attr, self.cls_edge.weight.expand(len(peptide_feat)-1, edge_attr.shape[-1])], 0)
        for l in self.convs:
            peptide_feat = l(peptide_feat, edge_index,
                             edge_attr=edge_attr) + peptide_feat
        return peptide_feat[:-1]


class FullModel(nn.Module):

    def __init__(self, args, dropout=0.1, return_attention=False):

        super().__init__()
        n_layers = args.n_layers
        d_model = args.d_model
        n_head = args.n_head
        d_k = args.d_k
        d_v = args.d_v
        d_inner = args.d_inner
        self.args = args
        self.peptide_model = ResidueGraphModel(args, d_model)
        self.repeated_module = RepeatedModule(edge_features, node_features, n_layers, d_model,
                                              n_head, d_k, d_v, d_inner, dropout=dropout)

        self.final_attention_layer = MultiHeadAttentionSequence(n_head, d_model,
                                                                d_k, d_v, dropout=dropout)

        self.final_ffn = FFN(d_model, d_inner, dropout=dropout)
        self.output_projection_nodes = nn.Linear(d_model, 2)

        self.softmax_nodes = nn.LogSoftmax(dim=-1)

        self.return_attention = return_attention
        self.out_layer = nn.Sequential(nn.Linear(72, 1), nn.Sigmoid())
        self.pro_pred = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.pep_pred = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, batch):
        peptide, protein, label, bert_features = batch
        peptide = peptide.to(self.args.device)
        protein = protein.to(self.args.device)
        bert_features = bert_features.to(self.args.device)
        pro_chaininfo = protein.pro_chaininfo
        neighbor_indices = protein.neighbor_indices
        nodes = protein.node_feat
        edges = protein.edge_feat
        pep_residue_index = peptide.residue_index[0].long()
        peptide_sequence = self.peptide_model(
            peptide.x, peptide.edge_index, peptide.edge_attr)  # [None]

        peptide_sequence = torch.scatter_reduce(peptide_sequence.new_zeros([pep_residue_index.max()+1, peptide_sequence.shape[-1]]), 0,  pep_residue_index[:, None].expand_as(peptide_sequence), peptide_sequence, include_self=False, reduce="mean")[None]
        peptide_sequence, nodes, edges, neighbor_indices, bert_features
        node_enc, sequence_enc, sequence_attention_list, graph_attention_list, \
            seq_graph_attention_list, graph_seq_attention_list = self.repeated_module(peptide_sequence,
                                                                                      nodes[None],
                                                                                      edges[None],
                                                                                      neighbor_indices[None], bert_features[None], pro_chaininfo)

        # sequence_enc=sequence_enc[0]
        seq_graph_attention = torch.cat(seq_graph_attention_list, 1).squeeze(0)
        graph_seq_attention = torch.cat(
            graph_seq_attention_list, 1).squeeze(0).transpose(-1, -2)
        pair_attn = torch.cat(
            [seq_graph_attention, graph_seq_attention], 0).permute(1, 2, 0)

        node_enc, final_node_seq_attention = self.final_attention_layer(
            node_enc, sequence_enc, sequence_enc)
        # sequence_enc=sequence_enc.squeeze(0)
        # sequence_enc=torch.scatter_reduce(sequence_enc.new_zeros([pep_residue_index.max()+1, sequence_enc.shape[-1]]), 0,  pep_residue_index[:, None].expand_as(sequence_enc), sequence_enc, include_self=False, reduce="mean")[None]
        sequence_enc = sequence_enc.squeeze(0)
        node_enc = node_enc.squeeze(0)
        merge_shape = [sequence_enc.shape[0],
                       node_enc.shape[0],  node_enc.shape[-1]]
        # torch.cat([node_enc[None].expand(merge_shape), sequence_enc[:,None].expand(merge_shape)], -1)).squeeze(-1)
        pred = self.out_layer(pair_attn).squeeze(-1)
        pro_pred = self.pro_pred(node_enc).squeeze(-1)
        pep_pred = self.pep_pred(sequence_enc).squeeze(-1)
        return [pred], [pro_pred], [pep_pred]
