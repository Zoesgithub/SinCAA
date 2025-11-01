from torch import nn
from .modules import *


class ReciprocalLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):

        super().__init__()

        self.sequence_attention_layer = MultiHeadAttentionSequence(n_head, d_model,
                                                                   d_k, d_v, dropout=dropout)

        self.graph_attention_layer = MultiHeadAttentionGraph(n_head, d_model,
                                                             d_k, d_v, dropout=dropout)

        self.reciprocal_attention_layer = MultiHeadAttentionReciprocal(n_head, d_model,
                                                                       d_k, d_v, dropout=dropout)

        self.ffn_seq = FFN(d_model, d_inner)

        self.ffn_graph = FFN(d_model, d_inner)

    def forward(self, sequence_enc, nodes, edges, pro_chaininfo):
        node_enc, graph_attention = self.graph_attention_layer(nodes, edges)
        seq_enc, sequence_attention =sequence_enc, None #self.sequence_attention_layer(
        #    sequence_enc, sequence_enc, sequence_enc)

        node_enc, seq_enc, node_seq_attention, seq_node_attention = self.reciprocal_attention_layer(node_enc,
                                                                                                    seq_enc,
                                                                                                    seq_enc,
                                                                                                    node_enc)
        node_enc = self.ffn_graph(node_enc)

        seq_enc = self.ffn_seq(seq_enc)

        return node_enc, seq_enc, graph_attention, sequence_attention, node_seq_attention, seq_node_attention
