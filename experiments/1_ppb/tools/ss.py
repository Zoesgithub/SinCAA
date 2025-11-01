import numpy as np
import pickle
import sys
from tqdm import tqdm

"""
From CAMP work, fitting for newest SSW algo.
"""

in_fasta = sys.argv[1]
in_msa = sys.argv[2]
out_mat = sys.argv[3]
name = sys.argv[4]


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


protein_vocabulary_dict = {}
f = open(in_fasta)  # peptide.fasta
i = 0
for line in tqdm(f.readlines()):
    if line[0] == ">":
        protein_vocabulary_dict[line[1:-1]] = i
        i += 1
pickle.dump(protein_vocabulary_dict, open(f"./{name}_dict.dict", "wb"))
f.close()
# print protein_vocabulary_dict.keys()
p_simi = np.zeros((len(protein_vocabulary_dict), len(protein_vocabulary_dict)))
print(p_simi.shape)
print(len(protein_vocabulary_dict))

count = 0
f = open(in_msa)  # peptide_output.txt
lines = f.readlines()
f.close()
print("total lines", len(lines))

for i in tqdm(range(len(lines))):
    try:
        if "target_name" in lines[i]:
            assert "optimal" in lines[i + 2]
            assert "sub" not in lines[i + 2]
            a = lines[i].strip("\n").split(":")[-1].strip()

            b = lines[i + 1].strip("\n").split(":")[-1].strip()
            c = float(int(lines[i + 2].strip("\n").split()[1]))
            # print(protein_vocabulary_dict[a], protein_vocabulary_dict[b])
            p_simi[protein_vocabulary_dict[a], protein_vocabulary_dict[b]] = c
    except Exception as e:
        print(e)
        print("wrong", i, a, b, c)
        # protein_vocabulary_dict[a], protein_vocabulary_dict[b])
        exit()

print(check_symmetric(p_simi))

for i in tqdm(range(len(p_simi))):
    for j in range(len(p_simi)):
        if i == j:
            continue
        p_simi[i, j] = p_simi[i, j] / (float(np.sqrt(p_simi[i, i]) * np.sqrt(p_simi[j, j])) + 1e-12)
for i in range(len(p_simi)):
    p_simi[i, i] = p_simi[i, i] / float(np.sqrt(p_simi[i, i]) * np.sqrt(p_simi[i, i]) + 1e-12)

print("p_simi", p_simi.shape)
print(check_symmetric(p_simi))
np.save(out_mat, p_simi)
