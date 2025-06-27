import sys
import random

load_path = sys.argv[1]

with open(load_path, "r") as f:
    content = f.readlines()

header = content[0].replace("smiles", "SMILES")

res = content[1:]
random.shuffle(res)
cut = int(len(res)*0.8)

with open("/".join(load_path.split("/")[:-1])+"/train_"+load_path.split("/")[-1], "w") as f:
    f.writelines(header)
    for line in res[:cut]:
        f.writelines(line)

with open("/".join(load_path.split("/")[:-1])+"/val_"+load_path.split("/")[-1], "w") as f:
    f.writelines(header)
    for line in res[cut:]:
        f.writelines(line)
