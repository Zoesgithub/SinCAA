import torch
from utils.rigid_utils import RobustRigid

@torch.enable_grad()
def relax(pred_cord, local_frame_indicator, mean_position_in_local, other_edges, lower_bound, num_iterations=20, eps=1e-8):
    assert other_edges.shape[1]==2 # other edges should be bidirectional
    mean_position_in_local=mean_position_in_local.to(pred_cord.dtype)
    lower_bound=lower_bound.to(pred_cord.dtype)
    
    def update_inner(X): # fix point method
        # handle local frames
        origin_position=X[local_frame_indicator[..., 0]]
        xaxis_position=X[local_frame_indicator[..., 1]]
        point_on_plane=X[local_frame_indicator[..., 2]]
        frame_transform=RobustRigid.from_3_points(xaxis_position, origin_position, point_on_plane)
        l=((X[local_frame_indicator[..., 3]]-frame_transform.apply(mean_position_in_local))**2).sum(-1).add(eps).sqrt().mean()#+((X-pred_cord)**2).sum(-1).add(eps).sqrt().mean()
        grad=torch.autograd.grad(l, X, retain_graph=True)[0] #torch.scatter_reduce(X.new_zeros(X.shape), 0,local_frame_indicator[..., 3][..., None].expand((local_frame_indicator.shape[0], 3)), ground_local_pos, reduce="sum",include_self=False)
        return X-grad#(fx_mean-X)/(grad-1)
    
    X=pred_cord
    assert not torch.isnan(X.sum()), X
    if not X.requires_grad:
        X.requires_grad=True
    for i in range(num_iterations):
        new_X=update_inner(X)
        if torch.isnan(new_X.sum()):
            print("Encounter nan")
            break
        X=new_X
    
    return X.to(pred_cord.dtype)
        
        