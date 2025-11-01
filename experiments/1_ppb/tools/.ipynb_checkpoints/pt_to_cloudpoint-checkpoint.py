from utils.utilities import write_to_pointcloud
import torch
import sys


file=sys.argv[1]
data=torch.load(file, weights_only=False)
write_to_pointcloud(data.mesh_vertex, [0.5 for _ in data.mesh_vertex], "tmp.pdb")
