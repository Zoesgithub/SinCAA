import pandas as pd
from rdkit import Chem
import os
import numpy as np
import random

from experiments.utils import load_model, get_emb_from_feat
from experiments.amino_acid import AminoAcid

import torch
import torch_geometric.nn as gnn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from loguru import logger
from tqdm import tqdm

from scipy.stats import pearsonr, spearmanr
from utils.data_utils import myHash
from copy import deepcopy


def load_data():
    data_path = "experiments/0_benchmark/cpp/CycPeptMPDB_Peptide_All.csv"
    data = pd.read_csv(data_path)
    return data


def load_repr(cache_path, model_path):
    if "nopretrain" in cache_path:
        if not os.path.exists(cache_path):
            nemb = []
            edges = []
            edge_attrs = []
            pseudo_emb = []
            all_smiles = []
            for smiles in tqdm(data["SMILES"]):
                feat = AminoAcid(aa_name=myHash(smiles), aa_idx=0,
                                 smiles=smiles).get_graph_with_gt()
                nemb.append(feat["nodes_int_feats"])
                edges.append(feat["edges"])
                edge_attrs.append(feat["edge_attrs"])
                all_smiles.append(smiles)
            torch.save(
                {
                    "node_embs": nemb,
                    "smiles": all_smiles,
                    "edges": edges,
                    "edge_attrs": edge_attrs
                },
                cache_path
            )

        else:
            cache_data = torch.load(cache_path, weights_only=False)
            nemb = cache_data["node_embs"]
            edges = cache_data["edges"]
            edge_attrs = cache_data["edge_attrs"]
            pseudo_emb = cache_data["node_embs"]
        return nemb, nemb, edges, edge_attrs

    if os.path.exists(cache_path):
        cache_data = torch.load(cache_path, weights_only=False)
        nemb = cache_data["node_embs"]
        edges = cache_data["edges"]
        edge_attrs = cache_data["edge_attrs"]
        pseudo_emb = cache_data["embs"]
        nemb = [torch.tensor(_).cuda() for _ in nemb]
    else:
        logger.info(f"Generating embedding from {model_path}")
        model = load_model(model_path)
        device = next(model.parameters()).device
        nemb = []
        edges = []
        edge_attrs = []
        pseudo_emb = []
        all_smiles = []
        for smiles in tqdm(data["SMILES"]):
            feat = AminoAcid(aa_name=myHash(smiles), aa_idx=0,
                             smiles=smiles).get_graph_with_gt()
            emb = get_emb_from_feat(feat, model, device)
            nemb.append(emb[0])
            edges.append(feat["edges"])
            edge_attrs.append(feat["edge_attrs"])
            pseudo_emb.append(emb[1])
            all_smiles.append(smiles)
        torch.save(
            {
                "embs": pseudo_emb,
                "node_embs": nemb,
                "smiles": all_smiles,
                "edges": edges,
                "edge_attrs": edge_attrs
            },
            cache_path
        )
    nemb = [torch.tensor(_).cuda() for _ in nemb]
    return nemb, pseudo_emb, edges, edge_attrs


class pepdata(Dataset):
    # index list is required for train val test spliting
    def __init__(self, index_list) -> None:
        super().__init__()
        self.index_list = index_list

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        index = self.index_list[index]
        node = nodes_embeds[index]
        if len(node.shape) == 3:
            node = node.squeeze(0)
        pseudo_node = pseudo_embeds[index]
        if isinstance(pseudo_node, list):
            pseudo_node = pseudo_node[0]
        if len(pseudo_node.shape) == 1:
            pseudo_node = pseudo_node[None]
        label = data["Permeability"][index]
        edge = edges[index]
        if edge.shape[0] == 2:
            edge = edge.transpose(1, 0)
        edge_attr = edge_attrs[index]
        if len(edge.shape) == 3:
            edge = edge.squeeze(0)
        if len(edge_attr.shape) == 3:
            edge_attr = edge_attr.squeeze(0)
        if edge_attr.shape[1] == 3:
            edge_attr = edge_attr.argmax(-1)[..., None]
        # concate node & edge
        if not isinstance(node, torch.Tensor):
            node = torch.tensor(node)
        if not isinstance(edge, torch.Tensor):
            edge = torch.tensor(edge)
        if not isinstance(edge_attr, torch.Tensor):
            edge_attr = torch.tensor(edge_attr)
        node = torch.cat([node, node.new_zeros([1, node.shape[-1]])], 0)
        redge = torch.stack(
            [torch.arange(len(node)-1), torch.zeros(len(node)-1)+len(node)-1], -1).long()
        ledge = torch.stack(
            [torch.zeros(len(node)-1)+len(node)-1, torch.arange(len(node)-1)], -1).long()
        edge = torch.cat(
            [edge, redge.to(edge.device), ledge.to(edge.device)], 0)
        edge_attr = torch.cat([edge_attr, edge_attr.new_zeros(
            [(len(node)-1)*2, *(edge_attr.shape[1:])])], 0)
        return {
            "edge": edge.long(),
            "edge_type": torch.tensor(edge_attr),
            "node": node,
            "label": torch.tensor(label).reshape(-1).float().clamp(-8, -4),
            "pseudo_node": torch.tensor(pseudo_node).float(),
        }


def collate_fn(batch):
    ret = {_: [] for _ in batch[0].keys()}
    num_node = 0
    batch_idx = []
    for i, v in enumerate(batch):
        for k in v:
            if k == "edge":
                ret[k].append(v[k]+num_node)
            else:
                ret[k].append(v[k])
        num_node += len(v["node"])
        batch_idx.append(torch.zeros(len(v["node"])).long()+i)
    ret = {k: torch.cat(ret[k], 0) for k in ret}
    ret["batch_index"] = torch.cat(batch_idx, 0)
    return ret


class GINModel(nn.Module):
    def __init__(self, num_layers, hidden_size=128) -> None:
        super().__init__()
        self.input_layer = nn.Linear(inp_size, hidden_size)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            l = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
            )
            self.convs.append(gnn.GINEConv(l))
        self.out_layer = nn.Linear(hidden_size, 1)
        self.edge_type_emb = nn.Embedding(100, hidden_size)
        if "nopretrain" in args.cache_path:
            self.node_emb = nn.ModuleList(
                [nn.Embedding(100, hidden_size) for _ in range(3)])
            self.input_layer = nn.Linear(hidden_size, hidden_size)
        else:
            self.node_emb = None

    def forward(self, data):
        if self.node_emb is not None:
            node = sum([self.node_emb[i](data["node"][..., i].long())
                       for i in range(data["node"].shape[-1])])/data["node"].shape[-1]
        else:
            node = data["node"].float()
        node = (self.input_layer(node))
        edge = data["edge"]
        batch_index = data["batch_index"]
        edge_attr = self.edge_type_emb(
            data["edge_type"].long()[..., :1].reshape(-1))
        assert edge_attr.shape[0] == edge.shape[0], f"{edge_attr.shape} {edge.shape}"
        assert edge.shape[-1] == 2, edge.shape
        edge = edge.transpose(1, 0)
        for l in self.convs:
            node = l(node, edge, edge_attr=edge_attr)+node
        agg_node_feat = gnn.pool.global_mean_pool(node, batch_index)
        return self.out_layer(agg_node_feat).squeeze(-1)


class LinearModel(nn.Module):
    def __init__(self, hidden_size=128) -> None:
        super().__init__()
        self.input_layer = nn.Linear(inp_size, hidden_size)
        self.out_layer = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, 1))
        if "nopretrain" in args.cache_path:
            self.node_emb = nn.ModuleList(
                [nn.Embedding(100, hidden_size) for _ in range(3)])
            self.input_layer = nn.Linear(hidden_size, hidden_size)
        else:
            self.node_emb = None

    def forward(self, data):
        if self.node_emb is not None:
            node = sum([self.node_emb[i](data["node"][..., i].long())
                       for i in range(data["node"].shape[-1])])/data["node"].shape[-1]
            node = torch.scatter_reduce(node.new_zeros(data["batch_index"].max().long().item(
            )+1, node.shape[-1]), 0, data["batch_index"][..., None].expand_as(node).long(), node, include_self=False, reduce="mean")
        else:
            node = data["pseudo_node"].float()
        node = (self.input_layer(node))

        return self.out_layer(node).squeeze(-1)


def main(args):
    num_epoch = 300
    lr = args.lr
    batch_size = 128
    data_path = os.path.join("experiments/0_benchmark/cpp", "data")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    save_path = os.path.join("experiments/0_benchmark/cpp", "data",
                             f"{args.exp_name}_{args.model_type}_{args.num_layers}")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        exit()
    seeds = [18291, 108921829, 2025]
    save_res = {}

    logger.add(os.path.join(save_path, "log"))

    for rep in range(3):
        test_pred_merge = []
        test_gt_merge = []
        random.seed(seeds[rep])
        np.random.seed(seeds[rep])
        torch.manual_seed(seeds[rep])
        torch.cuda.manual_seed(seeds[rep])
        torch.cuda.manual_seed_all(seeds[rep])

        for cv in range(3):
            train_index = np.load(
                f"experiments/0_benchmark/cpp/eval_index/Train_ID_cv{cv}.npy")-1
            val_index = np.load(
                f"experiments/0_benchmark/cpp/eval_index/Valid_ID_cv{cv}.npy")-1
            test_index = np.load(
                "experiments/0_benchmark/cpp/eval_index/Test_ID.npy")-1

            if "SMILES" not in save_res:
                save_res["SMILES"] = []
                for i in test_index:
                    save_res["SMILES"].append(data["SMILES"][i])
            train_data_loader = DataLoader(pepdata(
                train_index), collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
            val_data_loader = DataLoader(
                pepdata(val_index), collate_fn=collate_fn, batch_size=batch_size)
            test_data_loader = DataLoader(
                pepdata(test_index), collate_fn=collate_fn, batch_size=batch_size)
            if args.model_type == "G":
                pred_model = GINModel(args.num_layers, 256).cuda()
            else:
                pred_model = LinearModel(256).cuda()

            device = next(pred_model.parameters()).device
            optimizer = torch.optim.Adam(pred_model.parameters(), lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

            def move_data_to_device(d):
                return {k: d[k].to(device) for k in d}

            best_val_loss = float("inf")
            for epoch in range(num_epoch):
                train_loss = 0
                train_pred = []
                train_gt = []
                for d in train_data_loader:
                    pred_model.train()
                    optimizer.zero_grad()
                    d = move_data_to_device(d)
                    pred = pred_model(d)
                    label = d["label"]
                    assert pred.shape == label.shape, f"{pred.shape} {label.shape}"
                    loss = ((pred-label)**2).mean()
                    train_loss += loss.item()
                    train_pred.append(pred.detach().cpu().numpy())
                    train_gt.append(label.detach().cpu().numpy())
                    loss.backward()
                    optimizer.step()
                # scheduler.step()

                # evaluation
                val_pred = []
                val_label = []
                with torch.no_grad():
                    for d in val_data_loader:
                        pred_model.eval()
                        d = move_data_to_device(d)
                        val_pred.append(pred_model(d).detach().cpu().numpy())
                        val_label.append(d["label"].detach().cpu().numpy())
                val_loss = ((np.concatenate(val_pred, 0) -
                            np.concatenate(val_label, 0))**2).mean()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    test_pred = []
                    test_label = []
                    with torch.no_grad():
                        for d in test_data_loader:
                            pred_model.eval()
                            d = move_data_to_device(d)
                            test_pred.append(pred_model(
                                d).detach().cpu().numpy())
                            test_label.append(
                                d["label"].detach().cpu().numpy())
                    test_pred = np.concatenate(test_pred, 0)
                    test_label = np.concatenate(test_label, 0)
                    torch.save(pred_model.state_dict(), os.path.join(
                        save_path, f"{rep}_state_dict.pt"))
                    best_model = deepcopy(pred_model)
                    logger.info(
                        f"saving best in epoch {epoch}, test pr is {spearmanr(test_pred, test_label)} val loss {val_loss}")
            test_pred_merge.append(test_pred)
            test_gt_merge.append(test_label)
        test_all_pred = sum(test_pred_merge)/len(test_pred_merge)
        test_all_gt = sum(test_gt_merge)/len(test_gt_merge)
        logger.info(
            f"merge  test pr is {spearmanr(test_all_pred, test_all_gt)}")
        save_res[f"rep_{rep}_pred"] = test_all_pred
        save_res[f"rep_{rep}_gt"] = test_all_gt

    pd.DataFrame.from_dict(save_res).to_csv(
        os.path.join(save_path, "test.csv"))


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--model_type", choices=["L", "G"], required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--pretrain_model_path", type=str,
                        default="data/results/batch_n1_full_dl1/")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    data = load_data()
    nodes_embeds, pseudo_embeds,  edges, edge_attrs = load_repr(
        args.cache_path, args.pretrain_model_path)
    inp_size = nodes_embeds[0].shape[-1]
    main(args)
