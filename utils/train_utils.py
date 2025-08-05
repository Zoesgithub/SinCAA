import torch
from utils.data_utils import MolDataset, ChainDataset, collate_fn
from torch.utils.data import DataLoader
from models.sincaa import SinCAA
import torch.nn.functional as F
from loguru import logger
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np


def setup(rank, world_size):
    dist.init_process_group(
        "gloo", init_method="tcp://127.0.0.1:12359", rank=rank, world_size=world_size)


def inner_trainer(rank, world_size, args):
    if rank == 0:
        save_path = os.path.join(args.save_path, args.experiment_name)
        logger.add(os.path.join(save_path, "log"))

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
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        SinCAA(args).to(rank))
    model = DDP(model, device_ids=[
                rank], find_unused_parameters=True)

    if rank == 0:
        logger.info(f"The number of params:{model.module.get_num_params()}")
    logger.info("Finish set up, start run")

    train_data = ChainDataset(aa_path=args.train_aa_data_path,
                              mol_path=args.train_mol_data_path, cache_path=args.cache_path, world_size=world_size, rank=rank, max_combine=args.max_combine, istrain=True)

    valid_data = ChainDataset(aa_path=args.val_aa_data_path,
                              mol_path=args.val_mol_data_path,  cache_path=args.cache_path, world_size=world_size, rank=rank, max_combine=args.max_combine)
    train_data_loader = DataLoader(
        train_data, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, shuffle=True, drop_last=True)
    valid_data_loader = DataLoader(
        valid_data, batch_size=args.batch_size,  collate_fn=collate_fn, num_workers=args.num_workers, shuffle=False, drop_last=True)
    logger.info(
        f"train data size {len(train_data_loader)} val data size {len(valid_data_loader)}")

    save_path = os.path.join(args.save_path, args.experiment_name)
    save_example_path = os.path.join(save_path, "confs")
    if not os.path.exists(save_example_path) and rank == 0:
        os.mkdir(save_example_path)
    if os.path.exists(os.path.join(
            save_path, "model.statedict.best")):
        args.load_path = os.path.join(
            save_path, "model.statedict.best")
    start_epoch = 0
    if args.load_path is not None:
        print(f"loading from {args.load_path} ...")
        param = torch.load(args.load_path)
        model.load_state_dict(param["state_dict"], strict=False)
        start_epoch = param["epoch"]+1
        del param
    val_loss = 99999

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.num_epochs) ) * 0.5
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

    weight_contractive = args.cont_weight
    for epoch in range(start_epoch, args.num_epochs):
        print(optimizer.param_groups[0]['lr'])
        for i, d in enumerate(train_data_loader):
            optimizer.zero_grad()
            model.zero_grad()
            model.train()
            aa_data, mol_data, aa_neighbor_data, neg_data = d

            for Dict in [aa_data, mol_data, aa_neighbor_data, neg_data]:
                for k in Dict:
                    if hasattr(Dict[k], "cuda"):
                        Dict[k] = Dict[k].to(rank)
            rec_loss, aa_contrastive_loss, pad_acc = model.forward(
                aa_data, mol_data, aa_neighbor_data, neg_data, True)

            loss = aa_contrastive_loss*weight_contractive+rec_loss
            if args.aba:
                loss = rec_loss
            loss.backward()
            synchronize_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
            optimizer.step()

            if i % args.logger_step == 0 and rank == 0:
                logger.info(
                    f"epcoh {epoch} step {i} contrastive loss {aa_contrastive_loss.sum().item()} ; rec loss {rec_loss.item()} ; acc {pad_acc}")
            if i % 1000 == 0 and rank == 0:
                torch.save({"state_dict": model.state_dict(), "epoch": epoch}, os.path.join(
                    save_path, "model.statedict.tmp"))

        # scheduler.step()
        if True:
            logger.info(f"Finish training for epoch {epoch}")

            val_aa_con_loss = 0

            for i, d in enumerate(valid_data_loader):
                if args.aba:  # aba mode, eval with rec loss
                    model.train()
                else:
                    model.eval()
                with torch.no_grad():
                    aa_data, mol_data, aa_neighbor_data, neg_data = d

                    for Dict in [aa_data, mol_data, aa_neighbor_data, neg_data]:
                        for k in Dict:
                            if hasattr(Dict[k], "cuda"):
                                Dict[k] = Dict[k].to(rank)
                    rec_loss, aa_contrastive_loss, _ = model.forward(
                        aa_data, mol_data, aa_neighbor_data, neg_data,  True)

                    if args.aba:
                        val_aa_con_loss += rec_loss.item()
                    else:
                        val_aa_con_loss += rec_loss.item()

            all_loss = torch.tensor(val_aa_con_loss).to(rank)
            dist.all_reduce(all_loss, op=dist.ReduceOp.SUM)
            all_loss = all_loss.item()
            if rank == 0:
                logger.info(
                    f"Epoch {epoch} all loss {all_loss} ; contrastive loss {val_aa_con_loss} ;")

            if all_loss < val_loss and rank == 0:
                val_loss = all_loss
                torch.save({"state_dict": model.state_dict(), "epoch": epoch}, os.path.join(
                    save_path, "model.statedict.best"))
                logger.info(f"Saving best in Epoch {epoch}")


def trainer(args):
    world_size = torch.cuda.device_count()
    logger.info(f"training with {world_size} gpus")
    mp.spawn(inner_trainer, args=(world_size,  args),
             nprocs=world_size)
