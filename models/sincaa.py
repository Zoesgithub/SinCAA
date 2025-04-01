import torch.nn as nn
import torch_geometric.nn as gnn
import torch
from typing import Tuple
from utils.rigid_utils import RobustRigid
from utils.data_utils import collate_fn

def construct_gps(num_layers, channels, attn_type, num_head, norm="GraphNorm"):
    convs = nn.ModuleList()
    for _ in range(num_layers):
        net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )

        convs.append(gnn.GPSConv(channels, gnn.GINEConv(net), heads=num_head,
                                 attn_type=attn_type, norm=norm, act="gelu"))
    return convs

def construct_gin(num_layers, channels):
    convs = nn.ModuleList()
    for _ in range(num_layers):
        net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )

        convs.append(gnn.GINEConv(net))
    return convs

   

class SinCAA(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.topological_net = construct_gps(
            args.topological_net_layers, args.model_channels, num_head=args.num_head, attn_type="multihead", norm=args.norm)
        self.recover_info_convnet=construct_gps(1, args.model_channels, num_head=args.num_head, attn_type="multihead", norm=args.norm)[0]
        
        self.node_int_embeder = nn.ModuleList([nn.Embedding(
            100, args.model_channels) for _ in range(3)])
        self.edge_embeder = nn.ModuleList(
            [nn.Embedding(100,  args.model_channels)
             for _ in range(1)]
        )
        self.node_float_embeder = nn.Linear(
            4, args.model_channels)
        self.recovery_info=nn.Linear(args.model_channels, 300)
        self.softmax=nn.Softmax(-1)
        self.feat_dropout_rate=0.3
        self.out_similarity=nn.Sequential(nn.Linear(args.model_channels*2, 1), nn.Sigmoid())
        
    def get_num_params(self):
        total=sum(p.numel() for p in self.parameters())
        topological_net=sum(p.numel() for p in self.topological_net.parameters())
        return {"total":total, "topological_net":topological_net}

    def calculate_topol_emb(self, feats):
        node_int_feats = feats["nodes_int_feats"]
      
        node_float_feats = feats["nodes_float_feats"]
        edge_feats = feats["edge_attrs"]
        edge_index = feats["edges"]
        node_residue_index=feats["node_residue_index"]
        #=feats["pe"].long().reshape(-1)

        assert edge_index.shape[1] == edge_feats.shape[0], f"{edge_index.shape} {edge_feats.shape}"
        assert edge_index.shape[0] == 2, edge_index.shape
        assert node_int_feats.shape[-1]==3
        assert node_int_feats.max()<100, f"{node_int_feats.max}"

        # embedding
        node_residue_index = feats["node_residue_index"]
        
        node_int_emb_list = [self.node_int_embeder[i](
            node_int_feats[..., i]) for i in range(node_int_feats.shape[-1])]
        node_int_emb = sum(node_int_emb_list)/len(node_int_emb_list)
        node_float_emb = self.node_float_embeder(node_float_feats.float())
        node_emb = node_float_emb+node_int_emb
        
        edge_emb_list = [self.edge_embeder[i](
            edge_feats[..., i]) for i in range(edge_feats.shape[-1])]
        edge_emb = sum(edge_emb_list)/len(edge_emb_list)
        assert edge_emb.shape[0] == edge_index.shape[1]
        if self.training:
            mask=(torch.rand_like(node_emb[:, :1])<1-self.feat_dropout_rate).float()
        else:
            mask=torch.zeros_like(node_emb[:, :1])+1
        # gps forward
        x = node_emb*mask
        inpx=x
        assert edge_index.shape[-1]==0 or edge_index.max()<len(x), f"{edge_index.max} {x.shape}"
        
        for conv in self.topological_net:
            x = conv(x, edge_index, edge_attr=edge_emb,
                     batch=node_residue_index)
        
        ret_emb=torch.scatter_reduce(x.new_zeros(node_residue_index.max()+1, x.shape[-1]), 0, node_residue_index[..., None].expand_as(x), x, include_self=False, reduce="mean")
        recovery_info_loss=0
        for i in range(5):
            dmask=(torch.rand_like(node_emb[:, :1])<1-self.feat_dropout_rate).float()
            x_rep=x*dmask
            x_rep=self.recover_info_convnet(x_rep, edge_index, edge_attr=x_rep[edge_index[0]]+x_rep[edge_index[1]], batch=node_residue_index)
            recovery_info=self.recovery_info(x_rep).reshape(x.shape[0], 3, -1)[..., 0, :][dmask.squeeze(-1)<1]
            recovery_info_loss=recovery_info_loss+(nn.functional.cross_entropy(recovery_info, feats["nodes_int_feats"][..., 0][dmask.squeeze(-1)<1])).mean()
        return x, ret_emb, recovery_info_loss
    

    def inner_forward(self, data):
        node_emb, pseudo_emb, rec_loss = self.calculate_topol_emb(
            data)
        return  pseudo_emb, rec_loss
     

    def forward(self, aa_data, mol_data, neighbor_data):
        merge_feat=collate_fn([[aa_data], [mol_data], [neighbor_data]])[0]
        '''_, aa_emb, aa_rec_loss= self.calculate_topol_emb(aa_data)
        _, mol_emb, mol_rec_loss= self.calculate_topol_emb(mol_data)
        _, neigh_emb, neigh_rec_loss= self.calculate_topol_emb(neighbor_data)'''
        _, emb, rec_loss= self.calculate_topol_emb(merge_feat)
        na=aa_data["node_residue_index"].max()+1
        nm=mol_data["node_residue_index"].max()+1
        num_n=neighbor_data["node_residue_index"].max()+1
        aa_pseudo_emb=emb[:na]
        neighbor_pseudo_emb=emb[na+nm:]

        #return aa_emb,neigh_emb, aa_rec_loss+mol_rec_loss+neigh_rec_loss, self.out_similarity(torch.cat([aa_emb, neigh_emb], -1)).squeeze(-1)
        return aa_pseudo_emb,neighbor_pseudo_emb, rec_loss, self.out_similarity(torch.cat([aa_pseudo_emb, neighbor_pseudo_emb], -1)).squeeze(-1)
