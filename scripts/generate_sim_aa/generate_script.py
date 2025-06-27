import pandas as pd
binsize = 500000
data = pd.read_csv("data/AAList/clean_res.csv")
atom_num = data["atomnum"]
smiles = data["smiles"]
delta_atom_num = 2
obinsize = 500
with open("script/generate_sim_aa/merge_script.txt", "w") as f:
    for i in range(30):
        onum = len(list(smiles[(atom_num) == i]))
        ostart, oend = 0, obinsize
        while ostart < onum:
            ubond = min(30, i+delta_atom_num)
            anum = len(list(smiles[(
                atom_num <= ubond) & (atom_num >= i-delta_atom_num)]))
            start, end = 0, binsize
            while start < anum:
                f.writelines(
                    f"python -m Tools.build_sim_aa  --atom_num={i} --ostart={ostart} --oend={oend} --save_path data/tmp/res_{i}_{ostart}_{oend}_{start}_{end}.csv --start={start} --end={end}\n")
                start, end = end, end+binsize
            ostart = oend
            oend = oend+obinsize
