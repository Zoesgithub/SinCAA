from utils.data_utils import get_graph, myHash, collate_fn
from argparse import ArgumentParser, Namespace
from models.sincaa import SinCAA
import pandas as pd
import os
import torch
import json
import numpy as np
from experiments.utils import get_emb_from_feat
from tqdm import tqdm

def load_model(prefix):
    with open(prefix+"/config.json", "r") as f:
        args = Namespace(**json.load(f))
    model = SinCAA(args)

    cache_state_dict=torch.load(prefix+"model.statedict.best")["state_dict"]
    load_state_dict={k.replace("module.", ""):cache_state_dict[k] for k in cache_state_dict}
    
    mstate_dict=model.state_dict()
    for p in mstate_dict:
        if p not in load_state_dict:
            print("do not found", p)
            load_state_dict[p]=mstate_dict[p]
    keys=list(load_state_dict.keys())
    for p in keys:
        if p not in mstate_dict:
            print("more param", p)
            load_state_dict.pop(p)
    
    model.load_state_dict(load_state_dict)
    model=model.cuda()
    model.eval()
    return model

@torch.no_grad()
def main(args):
   
    model=load_model(args.pretrained_dir)
    data=pd.read_csv(args.csv_path)
    save_res={"embs":[], "node_embs":[], "smiles":[], "edges":[], "edge_attrs":[]}
    for smiles in tqdm(data["SMILES"]):
        
        if not isinstance(smiles, str):
            continue
        inp=get_graph(smiles=smiles)
        node_emb, emb=get_emb_from_feat(inp, model, "cuda")
        save_res["embs"].append(emb)
        save_res["node_embs"].append(node_emb)
        save_res["edges"].append(inp["edges"])
        save_res["edge_attrs"].append(inp["edge_attrs"])
        save_res["smiles"].append(smiles)
    torch.save(save_res, args.save_path)


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--csv_path", help="the path to input csv file, which should contain SMILES column", type=str, required=True)
    parser.add_argument("--pretrained_dir", help="the path to pretrained content", type=str, required=True)
    parser.add_argument("--save_path", help="the path to save path", type=str, default="data/aaemb/")
    args=parser.parse_args()
    
    main(args)