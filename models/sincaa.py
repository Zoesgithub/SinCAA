import torch.nn as nn
import torch_geometric.nn as gnn
import torch
from utils.data_utils import collate_fn
import random
from geomloss import SamplesLoss


def construct_gps(num_layers, channels, attn_type, num_head, norm="GraphNorm", num_inner_l=2):
    assert norm=="GraphNorm"
    convs = nn.ModuleList()
    for _ in range(num_layers):
        net = gnn.Sequential('x, edge_index, edge_attr', [
        (gnn.GINEConv(nn.Sequential(
            nn.Linear(channels, channels),
            nn.PReLU(),
            nn.Linear(channels, channels),
        )), 'x, edge_index, edge_attr -> x') for _ in range(num_inner_l)
       
    ])
        convs.append(gnn.GPSConv(channels, net, heads=num_head,
                                 attn_type=attn_type, norm=norm, act="PReLU"))
    return convs
   

class SinCAA(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        if args.decoder_layers>0:
            self.decoder=gnn.models.GIN(args.model_channels, args.model_channels, args.decoder_layers)
        else:
            self.decoder=None
        if not hasattr(args, "num_inner_l"):
            args.num_inner_l=2
        if hasattr(args, "model") and args.model=="GAT":
            self.topological_net=gnn.models.GAT(args.model_channels, args.model_channels, args.topological_net_layers)
            self.model="GAT"
        else:
            self.topological_net=construct_gps(args.topological_net_layers, args.model_channels, num_head=args.num_head, attn_type="multihead", norm=args.norm, num_inner_l=args.num_inner_l)
            self.model="GPS"
        assert args.norm=='GraphNorm'
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
        if self.aba==0:
            self.out_ln=nn.LayerNorm(args.model_channels)
      
        
        
    def get_num_params(self):
        total=sum(p.numel() for p in self.parameters())
        topological_net=sum(p.numel() for p in self.topological_net.parameters())
        return {"total":total, "topological_net":topological_net}
    
    def generate_mask(self, node_emb, edges, batch_id, dropout_rate=None, add_mask=False):
        assert len(node_emb.shape)==2, node_emb.shape
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
                if add_mask:
                    xs.append(x)
                else:
                    xs=x
            
                
        
        acc=0
        num_round=1
        for i in range(num_round):
            tx=x
            if self.decoder is not None:
                tx=self.decoder(tx, edge_index,)
            recovery_info=self.recovery_info(tx[mask.squeeze(-1)<1]).reshape(-1, 2, 100).reshape(-1, 100)
            l=feats["nodes_int_feats"][mask.squeeze(-1)<1][..., :2].reshape(-1)
            recovery_info_loss=recovery_info_loss+(nn.functional.cross_entropy(recovery_info, l,reduction='none' )).sum()/max(recovery_info.shape[0], 1)
            edge_recover_info=self.edge_recovery_info((tx[edge_index[0]]+tx[edge_index[1]])[edge_mask.squeeze(-1)<1]).reshape(-1, 2, 100).reshape(-1, 100)
            edge_l=edge_feats[edge_mask.squeeze(-1)<1].reshape(-1)
            recovery_info_loss=recovery_info_loss+(nn.functional.cross_entropy(edge_recover_info, edge_l, reduction='none' )).sum()/max(edge_recover_info.shape[0], 1)
            acc=acc+(recovery_info.argmax(-1)==l).float().sum()/max(l.shape[0],1)+(edge_recover_info.argmax(-1)==edge_l).float().sum()/max(edge_l.shape[0], 1)
        acc=acc/num_round/2
        return xs, recovery_info_loss, acc
    

    def inner_forward(self, data):
        node_emb, pseudo_emb, rec_loss = self.calculate_topol_emb(
            data)
        return  pseudo_emb, rec_loss
     

    def forward(self, aa_data, mol_data, neighbor_data, add_mask):
        assert add_mask
        emb_m, rec_loss_mol, mol_acc= self.calculate_topol_emb(mol_data, add_mask=add_mask)
        if self.aba==2: # use only zinc
            return rec_loss_mol, rec_loss_mol.new_zeros(rec_loss_mol.shape),mol_acc.item()
       
        if self.aba==1: # use both zinc and aa
            _, rec_loss_aa, aa_acc= self.calculate_topol_emb(aa_data, add_mask=True)
            return (rec_loss_mol+rec_loss_aa)/2, rec_loss_mol.new_zeros(rec_loss_mol.shape), mol_acc.item()
        merge_d=collate_fn([[aa_data], [neighbor_data]])[0]
        add_mask=True
        oemb_x, rec_loss_aa, aa_acc= self.calculate_topol_emb(merge_d, add_mask=add_mask)
        # local loss
        
        if isinstance(oemb_x, list):
            #emb_x=sum([_*self.generate_mask(oemb_x[0], merge_d['edges'], merge_d["batch_id"],dropout_rate=0.2, add_mask=True)[0] for _ in oemb_x])/len(oemb_x)
            emb_x=sum(oemb_x)/len(oemb_x)
        else:
            emb_x=oemb_x
        #emb_x=(emb_x-emb_x.mean(-1, keepdims=True))/emb_x.std(-1, keepdims=True).clamp(1e-8)
        #emb_x=self.out_ln(emb_x)
        counts = torch.bincount(merge_d["batch_id"])
        count_cumsum=torch.cumsum(counts, 0)-counts
        max_length=counts.max().item()+1
        num_batch=merge_d["batch_id"].max().item()+1
        proj_pos=merge_d["batch_id"]*max_length+torch.arange(merge_d["batch_id"].shape[0]).to(merge_d["batch_id"].device)-count_cumsum[merge_d["batch_id"]]
        batched_emb_x=emb_x.new_zeros([num_batch*max_length, emb_x.shape[-1]])
        batched_mask=emb_x.new_zeros([num_batch*max_length])
        batched_emb_x[proj_pos]=emb_x
        batched_mask[proj_pos]=1
        batched_emb_x=batched_emb_x.reshape([num_batch,max_length, emb_x.shape[-1]])
        batched_mask=batched_mask.reshape([num_batch,max_length])
        batched_mask=batched_mask/batched_mask.sum(-1, keepdims=True)
        loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.1)
        
        #emb_batch=torch.scatter_reduce(emb_x.new_zeros(merge_d["batch_id"].max().long().item()+1, emb_x.shape[-1]), 0, merge_d["batch_id"][..., None].expand_as(emb_x).long(), emb_x, include_self=False, reduce="mean")    
        #aa_pseudo_emb_batch=nn.functional.normalize(emb_batch[:len(emb_batch)//2])
        #neighbor_pseudo_emb_batch=nn.functional.normalize(emb_batch[len(emb_batch)//2:])
        #merge_sim=aa_pseudo_emb_batch@neighbor_pseudo_emb_batch.transpose(1,0)*5
        #label=torch.eye(merge_sim.shape[0]).to(merge_sim.device).float()
        tx=emb_x[merge_d["batch_id"]%2==0]
        ty=emb_x[merge_d["batch_id"]%2>0]
        tx=tx[torch.randperm(tx.size(0))[:200]]
        ty=ty[torch.randperm(ty.size(0))[:200]]

        neg=torch.cat([batched_emb_x[-1:],batched_emb_x[num_batch//2:-1]], 0)
        neg_mask=torch.cat([batched_mask[-1:],batched_mask[num_batch//2:-1]], 0)
        loss_neg=loss_fn(batched_mask[:num_batch//2], batched_emb_x[:num_batch//2],neg_mask, neg)
        loss_pos=loss_fn(batched_mask[:num_batch//2], batched_emb_x[:num_batch//2],batched_mask[num_batch//2:], batched_emb_x[num_batch//2:])
        contrast_loss=(loss_pos-loss_neg).clamp(-2).mean()+loss_neg.mean()*0.1#nn.functional.cross_entropy(merge_sim, label.argmax(-1))+loss_fn(tx, ty)*0.01
        
        '''threshold=-0.1
        if not add_mask:
            merge_sim=torch.einsum('ab,cb->ac', nn.functional.normalize(aa_pseudo_emb_batch), nn.functional.normalize(neighbor_pseudo_emb_batch))
            label=torch.eye(merge_sim.shape[0]).to(merge_sim.device).float()
            assert merge_sim.shape==label.shape
            contrast_loss=(merge_sim-merge_sim[label>0][..., None]).clamp(threshold).mean()
        else:
            merge_sim=torch.einsum('ab,cb->ac', nn.functional.normalize(aa_pseudo_emb_batch), nn.functional.normalize(neighbor_pseudo_emb_batch))
            label=torch.eye(merge_sim.shape[0]).to(merge_sim.device).float()
            assert merge_sim.shape==label.shape
            contrast_loss=(merge_sim-merge_sim[label>0][..., None]).clamp(threshold).mean()
            with torch.no_grad():
                temb_x, _, _= self.calculate_topol_emb(merge_d, add_mask=False)
            # consider local loss
            contrast_loss=contrast_loss+((oemb_x[-1]-temb_x)**2).sum(-1).add(1e-8).sqrt().mean()*0.01'''
        
        if add_mask:
            return (rec_loss_mol+rec_loss_aa)/2, contrast_loss, mol_acc.item() # minimize sim when dropout
        return rec_loss_mol, contrast_loss, mol_acc.item()
