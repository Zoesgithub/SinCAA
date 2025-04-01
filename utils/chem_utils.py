import numpy as np
from utils.rigid_utils import RobustRigid
import torch

def add_triangle_conn(edges):
    Dict = {}
    ret_frame = []
    for i in range(len(edges)):
        v1, v2 = edges[i]
        if v1 not in Dict:
            Dict[v1] = set()
        if v2 not in Dict:
            Dict[v2] = set()

        Dict[v1].add(v2)  # build all conn
    
    handled_root=set()
    handled_cor=set()
    root_queue=[list(Dict.keys())[0]] # origin points
    while len(root_queue) >0:
        k=root_queue.pop(0)
        if k in handled_root:
            continue
        handled_root.add(k)
        handled_cor.add(k)
        # find neighbors of k that have been used as origin as x axis
        for v in Dict[k]:
            if v in handled_cor:
                xaxis=v
                break
        else:
            xaxis=k
        
        # find neighbors of k that has not been used as origin as point on plane
        point_on_plane=None
        for v in Dict[k]:
            if v in handled_cor and v!=xaxis:
                point_on_plane=v
                break
                
        if point_on_plane is None:
            for v in Dict[k]:
                if v not in handled_cor:
                    point_on_plane=v
                    break
            else:
                point_on_plane=k
        for v in Dict[k]:
            if not v in handled_cor:
                ret_frame.append([k, xaxis, point_on_plane, v])
                handled_cor.add(v)
                root_queue.append(v)
    return np.array(ret_frame, dtype=int)

def pre_compute_distance_bound(default_position, radius, edges):
    edges=edges[edges[..., 0]<default_position.shape[0]]
    edges=edges[edges[..., 1]<default_position.shape[0]]
    
    local_frames=add_triangle_conn(edges)
    merge_edges=edges

    indicator=np.ones([default_position.shape[0], default_position.shape[0]])
    indicator[merge_edges[..., 0], merge_edges[..., 1]]=0
    indicator[merge_edges[..., 1], merge_edges[..., 0]]=0
    
    # build constraints about bond length and bond angle
    origin_position=default_position[local_frames[..., 0]]
    xaxis_position=default_position[local_frames[..., 1]]
    point_on_plane=default_position[local_frames[..., 2]]
    frame_transform=RobustRigid.from_3_points(torch.tensor(xaxis_position),torch.tensor(origin_position),torch.tensor(point_on_plane))
    mean_position_in_local_pos=frame_transform.invert_apply(torch.tensor(default_position[local_frames[..., 3]])).numpy()
    
    # build constraints about clash
    other_edges=np.stack(np.nonzero(indicator), -1) # the edges is bidirectional
    other_lower_bound=radius[other_edges[..., 0]]+radius[other_edges[..., 1]]-0.5
    
    return local_frames, mean_position_in_local_pos, other_edges, other_lower_bound
    
