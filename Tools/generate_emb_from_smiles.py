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
    '''for token in alphabet.all_toks:
        if len(token)==1:
            mol=Chem.MolFromSequence(token, flavor=0)
            if mol is not None:
                smiles=Chem.MolToSmiles(mol)
                if len(smiles)==0:
                    continue
                inp=get_graph(smiles=smiles)
                cpu_feats=collate_fn([[inp]])[0]
                inp_feats={k:cpu_feats[k].cuda() for k in cpu_feats}
                node_emb, pseudo_emb = model.calculate_topol_emb(
                    inp_feats, None)
                cpu_feats["node_emb"]=node_emb.detach().cpu()
                cpu_feats["pseudo_emb"]=pseudo_emb.detach().cpu()
                cpu_feats["label"]=torch.tensor(alphabet.get_idx(token)).reshape(-1)
                torch.save(cpu_feats, "data/std_emb/"+"/"+myHash(smiles))'''
    save_res=[]
    for smiles in tqdm(data["SMILES"]):
        
        if not isinstance(smiles, str):
            continue
        inp=get_graph(smiles=smiles)
        node_emb=get_emb_from_feat(inp, model, "cuda")[0]

        save_res.append(node_emb.mean(0).detach().cpu())
        #torch.save(cpu_feats, args.save_dir+"/"+myHash(smiles))
    save_res=np.stack(save_res)
    np.save(args.save_dir, save_res)


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--csv_path", help="the path to input csv file, which should contain SMILES column", type=str, required=True)
    parser.add_argument("--pretrained_dir", help="the path to pretrained content", type=str, required=True)
    parser.add_argument("--save_dir", help="the path to save path", type=str, default="data/aaemb/")
    args=parser.parse_args()
    
    main(args)