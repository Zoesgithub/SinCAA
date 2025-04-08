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
            nn.PReLU(),
            nn.Linear(channels, channels),
        )

        convs.append(gnn.GPSConv(channels, gnn.GINEConv(net), heads=num_head,
                                 attn_type=attn_type, norm=norm, act="PReLU"))
        #convs.append(gnn.GINEConv(net))
    return convs

def construct_gin(num_layers, channels):
    convs = nn.ModuleList()
    for _ in range(num_layers):
        net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.PReLU(),
            nn.Linear(channels, channels),
        )

        convs.append(gnn.GINEConv(net))
    return convs

   

class SinCAA(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        if hasattr(args, "model") and args.model=="GAT":
            self.topological_net=gnn.models.GAT(args.model_channels, args.model_channels, args.topological_net_layers)
            self.model="GAT"
        else:
            self.topological_net=construct_gps(args.topological_net_layers, args.model_channels, num_head=args.num_head, attn_type="multihead", norm=args.norm)
            self.model="GPS"
        #self.recover_info_convnet=gnn.models.GIN(args.model_channels, args.model_channels, 1)
        
        self.node_int_embeder = nn.ModuleList([nn.Embedding(
            100, args.model_channels) for _ in range(3)])
        self.edge_embeder = nn.ModuleList(
            [nn.Embedding(100,  args.model_channels)
             for _ in range(2)]
        )
        self.node_float_embeder = nn.Linear(
            4, args.model_channels)
        self.recovery_info=nn.Linear(args.model_channels, 200)
        self.feat_dropout_rate=0.4
        self.out_similarity=nn.Sequential(nn.Linear(args.model_channels*2, 1), nn.Sigmoid())
        self.transform_layer=nn.Linear(args.model_channels*args.topological_net_layers, args.model_channels)
        if not args.aba:
            self.emb_layer=nn.Linear(args.model_channels, 32)
            self.out_similarity=nn.Sequential(nn.Linear(32*2, 1), nn.Sigmoid())
        else:
            self.emb_layer=None
        
    def get_num_params(self):
        total=sum(p.numel() for p in self.parameters())
        topological_net=sum(p.numel() for p in self.topological_net.parameters())
        return {"total":total, "topological_net":topological_net}

    def calculate_topol_emb(self, feats, mask=None):
        node_int_feats = feats["nodes_int_feats"]
      
        node_float_feats = feats["nodes_float_feats"]
        edge_feats = feats["edge_attrs"]
        edge_index = feats["edges"]
        node_residue_index=feats["node_residue_index"]

        assert edge_index.shape[1] == edge_feats.shape[0], f"{edge_index.shape} {edge_feats.shape}"
        assert edge_index.shape[0] == 2, edge_index.shape
        assert node_int_feats.shape[-1]==3
        assert node_int_feats.max()<100, f"{node_int_feats.max}"

        node_int_emb_list = [self.node_int_embeder[i](
            node_int_feats[..., i]) for i in range(node_int_feats.shape[-1])]
        node_int_emb = sum(node_int_emb_list)/len(node_int_emb_list)
        node_float_emb = self.node_float_embeder(node_float_feats.float())
        node_emb = node_float_emb+node_int_emb
        
        edge_emb_list = [self.edge_embeder[i](
            edge_feats[..., i]) for i in range(edge_feats.shape[-1])]
        
        edge_emb = sum(edge_emb_list)/len(edge_emb_list)
        assert edge_emb.shape[0] == edge_index.shape[1]
            
        # gps forward
        x = node_emb
        inpx=x
        assert edge_index.max()==x.shape[0]-1
        assert edge_index.shape[-1]==0 or edge_index.max()<len(x), f"{edge_index.max} {x.shape}"
        xs=[]
        recovery_info_loss=0
        if self.model=="GAT":
            x=self.topological_net(x, edge_index,  edge_attr=edge_emb,batch=node_residue_index)

        else:
            for conv in self.topological_net:
                if self.training:
                    mask=(torch.rand_like(node_emb[:, :1])<1-self.feat_dropout_rate).float()
                else:
                    mask=torch.zeros_like(node_emb[:, :1])+1
                x=x*mask
                x = conv(x, edge_index, edge_attr=edge_emb,batch=node_residue_index)
                recovery_info=self.recovery_info(x[mask.squeeze(-1)<1]).reshape(-1, 2, 100).reshape(-1, 100)
                l=feats["nodes_int_feats"][..., :2][mask.squeeze(-1)<1].reshape(-1)
                recovery_info_loss=recovery_info_loss+(nn.functional.cross_entropy(recovery_info, l, reduce=False)).sum()/max(recovery_info.shape[0], 1)
                xs.append(x)
        x=self.transform_layer(torch.cat(xs, -1))
        ret_emb=torch.scatter_reduce(x.new_zeros(node_residue_index.max()+1, x.shape[-1]), 0, node_residue_index[..., None].expand_as(x), x, include_self=False, reduce="mean")
        if self.emb_layer is not None:
            ret_emb=self.emb_layer(ret_emb)
        recovery_info=self.recovery_info(x).reshape(-1, 2, 100).reshape(-1, 100)
        l=feats["nodes_int_feats"][..., :2].reshape(-1)
        recovery_info_loss=recovery_info_loss+(nn.functional.cross_entropy(recovery_info, l, reduce=False)).sum()/max(recovery_info.shape[0], 1)
        return x, ret_emb, recovery_info_loss, None
    

    def inner_forward(self, data):
        node_emb, pseudo_emb, rec_loss = self.calculate_topol_emb(
            data)
        return  pseudo_emb, rec_loss
     

    def forward(self, aa_data, mol_data, neighbor_data):
        '''merge_feat=collate_fn([[aa_data], [neighbor_data]])[0]
        merge_emb, emb, rec_loss, _= self.calculate_topol_emb(merge_feat)
        na=aa_data["node_residue_index"].max()+1
        aa_pseudo_emb=emb[:na]
        neighbor_pseudo_emb=emb[na:]'''
        mol_emb, _, mol_rec_loss, _=self.calculate_topol_emb(mol_data)
        aa_emb, aa_pseudo_emb, aa_rec_loss, _=self.calculate_topol_emb(aa_data)
        neighbor_emb, neighbor_pseudo_emb, neigh_rec_loss, _=self.calculate_topol_emb(neighbor_data)
        merge_emb=torch.cat([aa_emb, neighbor_emb, mol_emb], 0)
        #return aa_emb,neigh_emb, aa_rec_loss+mol_rec_loss+neigh_rec_loss, self.out_similarity(torch.cat([aa_emb, neigh_emb], -1)).squeeze(-1)
        return aa_pseudo_emb,neighbor_pseudo_emb, mol_rec_loss+aa_rec_loss+neigh_rec_loss, self.out_similarity(torch.cat([aa_pseudo_emb, neighbor_pseudo_emb], -1)).squeeze(-1), merge_emb
