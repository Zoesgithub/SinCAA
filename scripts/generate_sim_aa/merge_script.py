import sys
import os

prefix = sys.argv[1]

files = os.listdir("data/tmp/")

files = [_ for _ in files if _.startswith(prefix)]
write_res = {}
num_lines = None
sample_pos_num = 20
# group files
file_group = {}
for v in files:
    p = "_".join(v.split("_")[:4])
    if p not in file_group:
        file_group[p] = []
    file_group[p].append(v)
invalid = 0
for p in file_group:
    files = file_group[p]
    num_lines = None
    print(files)
    for idx, file in enumerate(files):
        with open("tmp/"+file, "r") as f:
            content = f.readlines()
        assert len(content) > 0
        assert num_lines is None or len(
            content) == num_lines, f"{len(content)} {num_lines}  {file}"
        num_lines = len(content)
        for line in content:
            source, target, similarity = line.replace("\n", "").split(",")
            target = target.split(";")
            similarity = [float(_) for _ in similarity.split(";")]
            assert len(target) == 20
            if not similarity[0] == 1:
                invalid += 1
                print(line, invalid)
                # continue
            # assert similarity[0]==1, line
            if source not in write_res:
                write_res[source] = []
            assert len(target) == len(similarity), f"{(target)} {(similarity)}"

            for a, b in zip(target, similarity):
                if b < 1:
                    write_res[source].append([a, b])
if len(write_res) > 0:
    with open("scripts/generate_sim_aa/merge_res/"+prefix.replace("/", "_")+".csv", "w") as f:
        for k in write_res:
            values = sorted(
                write_res[k], key=lambda x: -float(x[1]))[:sample_pos_num]
            f.writelines(
                f"{k},{';'.join([_[0] for _ in values])},{';'.join([str(_[1]) for _ in values])}\n"
            )
print(len(write_res))
