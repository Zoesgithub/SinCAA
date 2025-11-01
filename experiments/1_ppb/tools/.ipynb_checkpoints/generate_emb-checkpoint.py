from experiments.utils import get_emb_from_sdf, load_model
import os
import torch
from tqdm import tqdm
save_path="experiments/ppb/raw_data/vocab_emb/"
vocab_path="experiments/ppb/raw_data/vocab/"
if not os.path.exists(save_path):
    os.mkdir(save_path)
model=load_model().cuda()
device=next(model.parameters()).device
for name in tqdm(os.listdir(vocab_path)):
    if name.endswith("sdf"):
        name=name.split("_")[0]
        try:
            node_emb, pseudo_emb=get_emb_from_sdf(f"{name}_ideal", model, device, vocab_path)
            torch.save({
            "emb":node_emb.detach().cpu().numpy(),
            "pseudo_emb":pseudo_emb.detach().cpu().numpy()
            },os.path.join(save_path, name))
        except:
            continue