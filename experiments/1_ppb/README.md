# Experiments for Protein-Peptide Binding

First, download all relevant files from [PDB](https://www.rcsb.org/) and save them as `./raw_dir/*.cif`.  
The file containing cleaned protein-peptide binding pairs is located at `raw_data/make_data/peptide_chain.txt`.  

Next, prepare the data by following the instructions in `prework.ipynb`.  Then generate peptide embeddings with 
```bash
bash generate_peptide_emb.sh 0

Finally, run model training and evaluation using:

```bash
bash run.sh
