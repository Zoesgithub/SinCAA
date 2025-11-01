from rdkit import Chem
from utils.data_utils import get_graph, collate_fn
import torch
from models.sincaa import SinCAA
import torch
import json
import os
from argparse import Namespace
from utils.amino_acid import AminoAcid


def load_model(model_path):
    print(model_path)
    with open(os.path.join(model_path, "config.json"), "r") as f:
        content = json.load(f)
    args = Namespace(**content)
    if not hasattr(args, "decoder_layers"):
        args.decoder_layers = 2
    model = SinCAA(args).cuda()
    if os.path.exists(os.path.join(model_path, "model.statedict.best")):
        param = torch.load(os.path.join(model_path, "model.statedict.best"))
        print("loading from ", os.path.join(
            model_path, "model.statedict.best"))
    else:
        param = torch.load(os.path.join(model_path, "model.statedict.tmp"))
        print("loading from ", os.path.join(model_path, "model.statedict.tmp"))
    if "state_dict" in param:
        param = param["state_dict"]
    clean_param = {}
    for k in param:
        clean_param[k[7:]] = param[k]
    mstate_dict = model.state_dict()
    for p in mstate_dict:
        if p not in clean_param:
            print("do not found", p)
            clean_param[p] = mstate_dict[p]
    keys = list(clean_param.keys())
    for p in keys:
        if p not in mstate_dict:
            print("more param", p)
            clean_param.pop(p)
    model.load_state_dict(clean_param)
    model.eval()
    return model


@torch.no_grad()
def get_emb(smiles, model, device):
    graph = get_graph(smiles=smiles)
    data = collate_fn([[graph]])[0]
    data = {k: data[k].to(device) for k in data}
    node_emb, pseudo_emb, _ = model.calculate_topol_emb(data, add_mask=False)
    return node_emb, pseudo_emb


@torch.no_grad()
def get_emb_from_feat(feat, model, device):
    data = collate_fn([[feat]])[0]
    data["batch_id"] = data["node_residue_index"]
    assert data["batch_id"].max() == 0, data["batch_id"]
    # data=collate_fn([[feat]])[0]
    data = {k: data[k].to(device) if isinstance(
        data[k], torch.Tensor) else data[k] for k in data}
    node_emb, _, acc = model.calculate_topol_emb(data, add_mask=False)

    return node_emb, node_emb.mean(0)


@torch.no_grad()
def get_emb_from_sdf(sdf, model, device, ligand_path):
    aa = AminoAcid(sdf, 0, ligand_path)
    graph = aa.get_graph_with_gt()
    data = collate_fn([[graph]])[0]
    data = {k: data[k].to(device) for k in data}
    node_emb, pseudo_emb, _ = model.calculate_topol_emb(data, add_mask=False)
    ret_node_emb = node_emb.new_zeros(
        [max(aa._map_from_inner_idx_to_output_idx.keys())+1, node_emb.shape[-1]])
    for k in aa._map_from_inner_idx_to_output_idx:
        ret_node_emb[k] = node_emb[aa._map_from_inner_idx_to_output_idx[k]]
    return ret_node_emb, pseudo_emb
