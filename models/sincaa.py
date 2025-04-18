import torch.nn as nn
import torch_geometric.nn as gnn
import torch
from typing import Tuple
from utils.rigid_utils import RobustRigid
from utils.data_utils import collate_fn

def construct_gps(num_layers, channels, attn_type, num_head, norm="GraphNorm"):
    convs = nn.ModuleList()
    for _ in range(num_layers):
        net = gnn.Sequential('x, edge_index, edge_attr', [
        (gnn.GINEConv(nn.Sequential(
            nn.Linear(channels, channels),
            nn.PReLU(),
            nn.Linear(channels, channels),
        )), 'x, edge_index, edge_attr -> x'),
        (gnn.GINEConv(nn.Sequential(
            nn.Linear(channels, channels),
            nn.PReLU(),
            nn.Linear(channels, channels),
        )), 'x, edge_index, edge_attr -> x'),
       
    ])
       

        convs.append(gnn.GPSConv(channels, net, heads=num_head,
                                 attn_type=attn_type, norm=norm, act="PReLU"))
    return convs

def construct_gps_gin(num_layers, channels, attn_type, num_head, norm="GraphNorm"):
    convs = nn.ModuleList()
    for _ in range(num_layers):
        net = gnn.Sequential('x, edge_index', [
        (gnn.GINConv(nn.Sequential(
            nn.Linear(channels, channels),
            nn.PReLU(),
            nn.Linear(channels, channels),
        )), 'x, edge_index -> x'),
        (gnn.GINConv(nn.Sequential(
            nn.Linear(channels, channels),
            nn.PReLU(),
            nn.Linear(channels, channels),
        )), 'x, edge_index -> x'),
       
    ])

        convs.append(gnn.GPSConv(channels, net, heads=num_head,
                                 attn_type=attn_type, norm=norm, act="PReLU"))
    return convs

def construct_gin(num_layers, channels):
    convs = nn.ModuleList()
    for _ in range(num_layers):
        net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.PReLU(),
            nn.Linear(channels, channels),
        )

        convs.append(gnn.GINConv(net))
    return convs

def construct_gine(num_layers, channels):
    convs = nn.ModuleList()
    for _ in range(num_layers):
        net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.PReLU(),
            nn.Linear(channels, channels),
        )

        convs.append(gnn.GINConv(net))
    return convs

   

class SinCAA(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        if args.decoder_layers>0:
            self.decoder=gnn.models.GIN(args.model_channels, args.model_channels, args.decoder_layers)
        else:
            self.decoder=None
        if hasattr(args, "model") and args.model=="GAT":
            self.topological_net=gnn.models.GAT(args.model_channels, args.model_channels, args.topological_net_layers)
            self.model="GAT"
        else:
            self.topological_net=construct_gps(args.topological_net_layers, args.model_channels, num_head=args.num_head, attn_type="multihead", norm=args.norm)
            self.model="GPS"
        
        self.node_int_embeder = nn.ModuleList([nn.Embedding(
            100, args.model_channels) for _ in range(3)])
        self.edge_embeder = nn.ModuleList(
            [nn.Embedding(100,  args.model_channels)
             for _ in range(2)]
        )
        self.node_float_embeder = nn.Linear(
            4, args.model_channels)
        if args.norm=='BatchNorm':
            self.recovery_info=nn.Sequential(nn.BatchNorm1d(args.model_channels),nn.Linear(args.model_channels, 200)) 
            self.edge_recovery_info=nn.Sequential(nn.BatchNorm1d(args.model_channels),nn.Linear(args.model_channels, 200))
        else:
            self.recovery_info=nn.Linear(args.model_channels, 200)
            self.edge_recovery_info=nn.Linear(args.model_channels, 200) 
        self.feat_dropout_rate=0.5
        self.aba=args.aba
        print(self.aba)
        if not self.aba:
            self.out_similarity=nn.Sequential(nn.Linear(args.model_channels*2, 1), nn.Sigmoid())
            self.out_contrast=nn.Sequential(nn.Linear(args.model_channels*2, 1), nn.Sigmoid())
        
    def get_num_params(self):
        total=sum(p.numel() for p in self.parameters())
        topological_net=sum(p.numel() for p in self.topological_net.parameters())
        return {"total":total, "topological_net":topological_net}
    
    def generate_mask(self, node_emb, edges, batch_id, dropout_rate=None, add_mask=False):
        if dropout_rate is None:
            dropout_rate=self.feat_dropout_rate
        if add_mask:
            mask=(torch.rand_like(node_emb[:, :1])<1-dropout_rate).float()
        else:
            mask=torch.zeros_like(node_emb[:, :1])+1
        if edges is not None:
            edge_mask=((mask[edges[0]]+mask[edges[1]])==2).float()
        else:
            edge_mask=None
        return mask, edge_mask

    def calculate_topol_emb(self, feats, add_mask=False):
        node_int_feats = feats["nodes_int_feats"]
      
        node_float_feats = feats["nodes_float_feats"]
        edge_feats = feats["edge_attrs"]
        edge_index = feats["edges"]
      
        batch_id=feats["batch_id"].long()
      
       
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
        mask, edge_mask=self.generate_mask(x, edge_index, batch_id, add_mask=add_mask)
        x=x*mask
        inpx=x
        assert edge_index.max()==x.shape[0]-1, f"{edge_index.shape} {x.shape}"
        assert edge_index.shape[-1]==0 or edge_index.max()<len(x), f"{edge_index.max} {x.shape}"
        xs=[]
        recovery_info_loss=0
        #part_info=feats["part_info"]
        #dx_loss=0
        if self.model=="GAT":
            x=self.topological_net(x, edge_index,  edge_attr=edge_emb,batch=batch_id)

        else:
            for i,conv in enumerate(self.topological_net):
                assert len(x)==len(batch_id), f"{x.shape}{batch_id.shape}"
                assert len(edge_emb)==edge_index.shape[-1], f"{edge_emb.shape}{edge_index.shape}"
                if i==0:
                    edge_attr=edge_emb*edge_mask
                else:
                    edge_attr=x[edge_index[0]]+x[edge_index[1]]
                x = conv(x, edge_index, edge_attr=edge_attr,batch=batch_id)
            
                
                
        ret_emb=torch.scatter_reduce(x.new_zeros(feats["batch_id"].max().long().item()+1, x.shape[-1]), 0, feats["batch_id"][..., None].expand_as(x).long(), x, include_self=False, reduce="sum")
        acc=0
        num_round=1
        for i in range(num_round):
            #emask, eedge_mask=self.generate_mask(x, edge_index,batch_id, 0.3)
            tx=x#*emask
            #edge_mask=eedge_mask*edge_mask
            #mask=emask*mask
            if self.decoder is not None:
                tx=self.decoder(tx, edge_index,)
            recovery_info=self.recovery_info(tx[mask.squeeze(-1)<1]).reshape(-1, 2, 100).reshape(-1, 100)
            l=feats["nodes_int_feats"][mask.squeeze(-1)<1][..., :2].reshape(-1)
            recovery_info_loss=recovery_info_loss+(nn.functional.cross_entropy(recovery_info, l, reduce=False)).sum()/max(recovery_info.shape[0], 1)
            edge_recover_info=self.edge_recovery_info((tx[edge_index[0]]+tx[edge_index[1]])[edge_mask.squeeze(-1)<1]).reshape(-1, 2, 100).reshape(-1, 100)
            edge_l=edge_feats[edge_mask.squeeze(-1)<1].reshape(-1)
            recovery_info_loss=recovery_info_loss+(nn.functional.cross_entropy(edge_recover_info, edge_l, reduce=False)).sum()/max(edge_recover_info.shape[0], 1)
            acc=acc+(recovery_info.argmax(-1)==l).float().sum()/max(l.shape[0],1)+(edge_recover_info.argmax(-1)==edge_l).float().sum()/max(edge_l.shape[0], 1)
        acc=acc/num_round/2
        return x, ret_emb, recovery_info_loss, acc
    

    def inner_forward(self, data):
        node_emb, pseudo_emb, rec_loss = self.calculate_topol_emb(
            data)
        return  pseudo_emb, rec_loss
     

    def forward(self, aa_data, mol_data, neighbor_data, add_mask):
        assert add_mask
        _, aa_pseudo_emb, rec_loss_aa, aa_acc= self.calculate_topol_emb(aa_data, add_mask=True)
        _, _, rec_loss_mol, mol_acc= self.calculate_topol_emb(mol_data, add_mask=add_mask)
        if self.aba:
            return aa_pseudo_emb, aa_pseudo_emb, (rec_loss_mol+rec_loss_aa)/2, rec_loss_aa.new_zeros(len(aa_pseudo_emb)), (mol_acc).item(), rec_loss_aa.new_zeros(len(aa_pseudo_emb))
        else:
            #neighbor_data["batch_id"]=neighbor_data["node_residue_index"]
            _, neighbor_pseudo_emb, rec_loss_n, n_acc= self.calculate_topol_emb(neighbor_data, add_mask=True)
            contract_pos=self.out_contrast(torch.cat([aa_pseudo_emb, neighbor_pseudo_emb], -1)).squeeze(-1)
            expand_shape=(aa_pseudo_emb.shape[0], neighbor_pseudo_emb.shape[0], neighbor_pseudo_emb.shape[-1])
            contract_neg=self.out_contrast(torch.cat([aa_pseudo_emb[:, None].expand(expand_shape), aa_pseudo_emb[None].expand(expand_shape)], -1)).squeeze(-1)
            contract_neg=contract_neg[torch.eye(contract_neg.shape[0]).to(contract_neg.device)==0]
            contrast_loss=torch.log(contract_pos.clamp(1e-8)).mean()+torch.log((1-contract_neg).clamp(1e-8)).mean()
            return aa_pseudo_emb,neighbor_pseudo_emb, (rec_loss_mol+rec_loss_aa+rec_loss_n)/3, self.out_similarity(torch.cat([aa_pseudo_emb, neighbor_pseudo_emb], -1)).squeeze(-1), (mol_acc).item(), -contrast_loss
