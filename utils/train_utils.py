import torch
from utils.data_utils import MolDataset, collate_fn
from torch.utils.data import DataLoader, DistributedSampler
from models.sincaa import SinCAA
import torch.nn.functional as F
from loguru import logger
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from utils.align_utils import get_optimal_align_by_residue
from utils.rigid_utils import RobustRigid
from copy import deepcopy
import numpy as np

def setup(rank, world_size):
    dist.init_process_group(
        "gloo", init_method="tcp://127.0.0.1:12359", rank=rank, world_size=world_size)

    
def get_constraints(pred_cord, local_frame_indicator, mean_position_in_local, eps=1e-6):
    if len(pred_cord.shape)==2:
        tpred_cord=pred_cord[None]
    else:
        tpred_cord=pred_cord
    ret=0
    for i in range(tpred_cord.shape[0]):
        pred_cord=tpred_cord[i]
        origin_position=pred_cord[local_frame_indicator[..., 0]]
        xaxis_position=pred_cord[local_frame_indicator[..., 1]]
        point_on_plane=pred_cord[local_frame_indicator[..., 2]]
        frame_transform=RobustRigid.from_3_points(xaxis_position, origin_position, point_on_plane)
        ground_local_pos=frame_transform.apply(mean_position_in_local)
        ret=ret+((ground_local_pos-pred_cord[local_frame_indicator[..., 3]])**2).sum(-1).add(eps).sqrt().mean()
    return ret/tpred_cord.shape[0]


def l2_loss(pred, gt, node_residue_index,node_is_pseudo, permutation,  eps=1e-6, perm_threshold=0.1, inf=10000000):
    if pred.shape==gt.shape:
        pred=pred[None]
    
    tpred=pred
    node_is_pseudo_mask=pred.new_ones(pred.shape[-2])
    node_is_pseudo_mask[node_is_pseudo]=0
    batch_size=node_residue_index.max()+1
    GT=[]
    PRED=[]
    RES=[]
    PERM=[]
    for i in range(tpred.shape[0]):
        pred=tpred[i]
        assert pred.shape == gt.shape, f"{pred.shape} {gt.shape}"    
        RES.append( node_residue_index[node_is_pseudo_mask>0]+batch_size*i)
        GT.append(gt[node_is_pseudo_mask>0])
        PRED.append(pred[node_is_pseudo_mask>0])
        PERM.append(permutation[node_is_pseudo_mask>0])
      
    
    loss=[]
    PERM=torch.cat(PERM, 0)
    GT=torch.cat(GT, 0)
    RES=torch.cat(RES, 0)
    PRED=torch.cat(PRED, 0)

    for i in range(PERM.shape[-1]):
        o=PERM[..., i]
        if o.max()==-1:
            break
        assert RES.min()>-1, RES
        non_mask_pos=(o<0).float()
        o[o<0]=PERM[..., 0][o<0]
      
        l=get_optimal_align_by_residue(GT[o],PRED,RES, eps=eps)
     
        loss.append(l)#+(1-m)*inf)
        
    loss=torch.stack(loss, -1)
    return loss.min(-1).values.mean()
    
def update(student, teacher):
    with torch.no_grad():
        m = 0.996
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

def contrastive_loss(pred, pos,eps=1e-6, rescale_factor=10,pmask=None):
    # get_distance
    pred=torch.nn.functional.normalize(pred)
    pos=torch.nn.functional.normalize(pos)
    
    assert pred.shape == pos.shape, f"{pred.shape}{pos.shape}"
    pos_distance = (pred*pos).sum(-1, keepdim=True)*rescale_factor  # should be maximized
    neg_distance = torch.einsum('ab,cb->ac', pred, pred)*rescale_factor
    mask = torch.zeros_like(neg_distance)
    mask[torch.arange(len(pred)), torch.arange(len(pred))]=rescale_factor
    mask[pmask>0]=rescale_factor
    neg_distance = (neg_distance*(mask<rescale_factor).float()-mask)
    
    label = pos_distance.new_zeros(pos_distance.shape[0]).long()
    pred = torch.cat([pos_distance, neg_distance], 1)
    #loss = (neg_distance-pos_distance).clamp(-1).mean()#
    loss=F.cross_entropy(pred, label, reduce=False)
    acc = pred.argmax(-1) == 0
    
    
    return loss.mean(), acc


def inner_trainer(rank, world_size, args):
    if rank==0:
        save_path=os.path.join(args.save_path, args.experiment_name)
        logger.add(os.path.join(save_path, "log"))
        
    def pairwise_sync(tensor):
        gathered_embeddings = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_embeddings, tensor)
        gathered_embeddings.pop(rank)
        gathered_embeddings=gathered_embeddings+[tensor]
        return torch.cat(gathered_embeddings, dim=0)
        
        
    def synchronize_gradients(model):
        world_size = dist.get_world_size()
        if world_size == 1:
            return 
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size 
    logger.info(f"Start setting up {rank}/{world_size}")
    setup(rank, world_size)
    logger.info(f"Start init {rank}/{world_size}")
    model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(SinCAA(args).to(rank))
    model = DDP(model, device_ids=[
                rank], find_unused_parameters=True)
    
    if rank==0:
        logger.info(f"The number of params:{model.module.get_num_params()}")
    logger.info("Finish set up, start run")
    
    
    
    train_data = MolDataset(aa_path=args.train_aa_data_path,
                            mol_path=args.train_mol_data_path, cache_path=args.cache_path,world_size=world_size, rank=rank, num_level=args.max_level)
    
    valid_data = MolDataset(aa_path=args.val_aa_data_path,
                            mol_path=args.val_mol_data_path,  cache_path=args.cache_path,world_size=world_size, rank=rank, num_level=args.max_level)
    train_data_loader = DataLoader(
        train_data, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, shuffle=True, drop_last=True)
    valid_data_loader = DataLoader(
        valid_data, batch_size=args.batch_size,  collate_fn=collate_fn, num_workers=args.num_workers, shuffle=False, drop_last=True)
    logger.info(f"train data size {len(train_data_loader)} val data size {len(valid_data_loader)}")
    
    save_path = os.path.join(args.save_path, args.experiment_name)
    save_example_path = os.path.join(save_path, "confs")
    if not os.path.exists(save_example_path) and rank == 0:
        os.mkdir(save_example_path)
    if os.path.exists(os.path.join(
                    save_path, "model.statedict.best")):
        args.load_path=os.path.join(
                    save_path, "model.statedict.best")
    start_epoch=0
    if args.load_path is not None:
        print(f"loading from {args.load_path} ...")
        param=torch.load(args.load_path)
        model.load_state_dict(param["state_dict"])
        del param
    val_loss = 99999
    train_map_between_neighbors=train_data.build_neighbor_key()
    val_map_between_neighbors=valid_data.build_neighbor_key()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    #scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.num_epochs) ) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    def get_neighbor_mask(iindex,jindex, map_dict):
        ret=torch.zeros([len(iindex), len(jindex)])
        for i in range(len(ret)):
            for j in range(len(jindex)):
                if i in map_dict[j] or j in map_dict[i]:
                    ret[i,j]=1
        return ret
    model_ema=deepcopy(model)
    for epoch in range(start_epoch,args.num_epochs):
        print(scheduler.get_last_lr())
        for i, d in enumerate(train_data_loader):
            optimizer.zero_grad()
            model.train()
            aa_data, mol_data, aa_neighbor_data = d

            for Dict in [aa_data, mol_data, aa_neighbor_data]:
                for k in Dict:
                    if hasattr(Dict[k], "cuda"):
                        Dict[k] = Dict[k].to(rank)
            aa_pseudo_emb, neighbor_pseudo_emb, rec_loss, similarity, new_emb = model.forward(
                aa_data, mol_data, aa_neighbor_data)
            with torch.no_grad():
                model_ema.eval()
                old_aa_pseudo_emb, _, _, _, old_emb=model_ema.forward(aa_data, mol_data, aa_neighbor_data, mask=None)
            st_loss= ((1 - ( torch.nn.functional.normalize(old_emb, p=2, dim=-1) *  torch.nn.functional.normalize(new_emb, p=2, dim=-1)).sum(dim=-1)).pow_(3)).mean()
            # reduce to one device
            all_aa_pseudo_emb=aa_pseudo_emb
            all_neighbor_pseudo_emb=neighbor_pseudo_emb
            all_neighbor_index=aa_data["index"]
            aa_contrastive_loss, acc = contrastive_loss(
                all_aa_pseudo_emb, all_neighbor_pseudo_emb,  pmask=get_neighbor_mask(all_neighbor_index,all_neighbor_index,train_map_between_neighbors))
            assert aa_data["sim"].shape==similarity.shape
            similarity_loss=-(torch.log(similarity)*aa_data["sim"]+torch.log(1-similarity)*(1-aa_data["sim"])).mean()
            
            loss =aa_contrastive_loss+rec_loss+similarity_loss+st_loss
            if args.aba:
                loss=rec_loss+similarity_loss+st_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            synchronize_gradients(model)
            optimizer.step()
            if torch.isnan(loss):
                print(aa_pred_cord, aa_data_gt, aa_pseudo_emb)
                exit()
            if i % args.logger_step == 0 and rank==0:
                logger.info(
                    f"epcoh {epoch} step {i} contrastive loss {aa_contrastive_loss.item()} ;  train acc { acc.float().sum().item()/len(acc)} ; rec loss {rec_loss.item()} ; sim loss {similarity_loss.item()} ; st loss {st_loss.item()}")
            update(model, model_ema)
        scheduler.step()
        if epoch%5==4:
            logger.info(f"Finish training for epoch {epoch}")
            val_aa_l2_loss = 0
            val_mol_l2_loss = 0
            val_aa_con_loss = 0
            val_acc = 0
            val_num = 0

            for i, d in enumerate(valid_data_loader):
                model.eval()
                with torch.no_grad():
                    aa_data, mol_data, aa_neighbor_data = d

                    for Dict in [aa_data, mol_data, aa_neighbor_data]:
                        for k in Dict:
                            if hasattr(Dict[k], "cuda"):
                                Dict[k] = Dict[k].to(rank)
                    aa_pseudo_emb, neighbor_pseudo_emb, rec_loss, similarity, _= model.forward(
                        aa_data, mol_data, aa_neighbor_data)

                    all_aa_pseudo_emb=aa_pseudo_emb#pairwise_sync(aa_pseudo_emb)      
                    all_neighbor_pseudo_emb=neighbor_pseudo_emb#pairwise_sync(neighbor_pseudo_emb)      
                    all_neighbor_index=aa_data["index"]#pairwise_sync(aa_data["index"])
                    aa_contrastive_loss, acc = contrastive_loss(
                    all_aa_pseudo_emb, all_neighbor_pseudo_emb,  pmask=get_neighbor_mask(all_neighbor_index,all_neighbor_index,train_map_between_neighbors)) 
                    similarity_loss=-(torch.log(similarity)*aa_data["sim"]+torch.log(1-similarity)*(1-aa_data["sim"])).mean()
                    if args.aba:
                        val_aa_con_loss+=rec_loss.item()
                    else:
                        val_aa_con_loss += aa_contrastive_loss.item()+similarity_loss.item()
                    val_acc += acc.float().sum().item()
                    val_num += acc.shape[0]

            all_loss =  torch.tensor(val_aa_con_loss).to(rank)
            dist.all_reduce(all_loss, op=dist.ReduceOp.SUM)
            all_loss = all_loss.item() 
            if rank==0:
                logger.info(
                f"Epoch {epoch} all loss {all_loss} aa cord loss {val_aa_l2_loss} ; mol cord loss {val_mol_l2_loss} ; contrastive loss {val_aa_con_loss} ; acc {val_acc/val_num}")
            if all_loss < val_loss and rank == 0:
                val_loss = all_loss
                torch.save({"state_dict":model.state_dict(), "epoch":epoch}, os.path.join(
                    save_path, "model.statedict.best"))
                logger.info(f"Saving best in Epoch {epoch}")


def trainer(args):
    world_size = torch.cuda.device_count()
    logger.info(f"training with {world_size} gpus")
    mp.spawn(inner_trainer, args=(world_size,  args),
             nprocs=world_size)
