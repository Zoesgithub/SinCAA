from utils.similarity_utils import sample_normed_pos, count_distance, get_non_empty_coord
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from multiprocessing import Pool
import time
import heapq
from tqdm import tqdm


def get_pos_coarse(s, grid_length=2):
    try:
        ret = sample_normed_pos(s, 5, grid_length=grid_length)
    except:
        return 1
    return ret


def get_pos(s):
    try:
        ret = sample_normed_pos(s, 20, grid_length=0.5)
    except:
        return 1
    
    return ret

def pre_pos(s):
    try:
        ret = sample_normed_pos(s, 20, grid_length=0.5, return_res=False)
    except:
        return 1
    
    return ret

def pre_coarse_pos(s):
    try:
        ret = sample_normed_pos(s, 5, grid_length=1, return_res=False)
    except:
        return 1
    
    return ret


def get_distance(a, b, threshold):
    smiles, b = b
    try:
        d = count_distance(a, b, threshold=threshold)
    except:
        return [0, smiles]

    return [d, smiles]

def one_step(inps, tolerance, distance=None):
    if distance is None:
        distance=[]
  
    d=get_distance(*inps, threshold=-2)
    
    if len(distance)>=tolerance and d[0]>distance[0][0]:
        heapq.heappush(distance, [float(d[0]), d[1]])
        heapq.heappop(distance)[0]
    else:
        heapq.heappush(distance, [float(d[0]), d[1]])
    return distance

def pre_process(save_path, candidate_smiles, candidate_target_smiles, step=1000):
    # filter with grid_length=2
    coarse_rep_of_source={s: get_pos_coarse(s, 2) for s in candidate_smiles}
    ret_res={s:[] for s in candidate_smiles}
    for t in tqdm(candidate_target_smiles):
        t_pos=get_pos_coarse(t, 2)
        for s in ret_res:
            one_step([coarse_rep_of_source[s], [t, t_pos]], topk_tolerance*5,  ret_res[s])
    map_from_t_to_s={}
    for k in ret_res:
        for v in ret_res[k]:
            if v[1] not in map_from_t_to_s:
                map_from_t_to_s[v[1]]=[]
            map_from_t_to_s[v[1]].append(k)
            
    # filter with grid_length=1
    coarse_rep_of_source={s: get_pos_coarse(s, 1) for s in candidate_smiles}
    ret_res={s:[] for s in candidate_smiles}
    for t in tqdm(map_from_t_to_s):
        t_pos=get_pos_coarse(t, 1)
        for s in map_from_t_to_s[t]:
            one_step([coarse_rep_of_source[s], [t, t_pos]], topk_tolerance, ret_res[s])
   
    with open(save_path, "w") as f:
        for s in ret_res:
            distance=ret_res[s]
            sorted_distance = sorted(distance, key=lambda x: -float(x[0]))[:topk_tolerance]
            f.writelines(
                f"{s},{';'.join([_[1] for _ in sorted_distance])},{';'.join([str(_[0]) for _ in sorted_distance])}\n")
            f.flush()
            
def main(args):
    if os.path.exists(args.save_path):
        return 
    f=open(args.save_path, "w")
    f.close()
    data = pd.read_csv(args.data_path)
    atom_num = data["atomnum"]
    smiles = data["smiles"]
    candidate_smiles = list(smiles[atom_num == args.atom_num])[args.ostart:args.oend]
    ubond=min(args.atom_num+delta_atom_num, 30)
    candidate_target_smiles = list(smiles[(
        atom_num <= ubond) & (atom_num >= args.atom_num-delta_atom_num)])
    
    num_process=10
    pool = Pool(num_process)
    
    pool.map(
        pre_pos, candidate_target_smiles[args.start:args.end])
    pool.map(
        pre_coarse_pos, candidate_target_smiles[args.start:args.end])
    
    pool.close()
    pool.join()
    stime=time.time()
    
    print(
        f"handling {len(candidate_smiles)} with {len(candidate_target_smiles[args.start:args.end])} neighbors {args.save_path} time in handling position {time.time()-stime}")
    pre_process(args.save_path, candidate_smiles, candidate_target_smiles[args.start:args.end])
    
    with open(args.save_path, "r") as f:
        preprocess_res=f.readlines()
    # get all neighbors
    print("finish preprocess for", args.save_path)
    map_from_targe_to_s={}
    source=[]
    for line in preprocess_res:
        smiles, target_smiles=line.split(",")[:2]
        target_smiles=target_smiles.split(";")
        source.append(smiles)
        for t in target_smiles:
            if t not in map_from_targe_to_s:
                map_from_targe_to_s[t]=[]
            map_from_targe_to_s[t].append(smiles)
    res_dict={}
    source_pos={_:get_pos(_) for _ in source}
    print(len(map_from_targe_to_s), len(source_pos), len(target_smiles))
    for t in tqdm(map_from_targe_to_s.keys()):
        t_pos=get_pos(t)
        for s in map_from_targe_to_s[t]:
            d=get_distance(source_pos[s], [t,t_pos], threshold=-1)
            if s not in res_dict:
                res_dict[s]=[]
            res_dict[s].append(d)
    with open(args.save_path, "w") as f:
        for k in (res_dict):
            distance=res_dict[k]
            sorted_distance = sorted(distance, key=lambda x: -x[0])
            f.writelines(
                f"{k},{';'.join([_[1] for _ in sorted_distance[:sample_pos_num]])},{';'.join([str(_[0]) for _ in sorted_distance[:sample_pos_num]])}\n")
            f.flush()
            
    print("finish ", args.save_path)
            

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", help="the path to mol data",
                        default="data/AAList/clean_res.csv")
    parser.add_argument(
        "--atom_num", help="the number of atoms to handle", type=int)
    parser.add_argument(
        "--start", help="start position", type=int)
    parser.add_argument(
        "--end", help="end position", type=int)
    parser.add_argument(
        "--ostart", help="ostart position", type=int)
    parser.add_argument(
        "--oend", help="oend position", type=int)
    parser.add_argument("--save_path", help="the path to save results")
    args = parser.parse_args()
    
    sample_pos_num = 20
    delta_atom_num = 2
    topk_tolerance=60
    main(args)
