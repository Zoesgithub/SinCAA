# Experiments for Cell Penetration Prediction

First, download the data from [CycPeptMP](https://github.com/akiyamalab/cycpeptmp) and place the following files in the current directory:  

- `CycPeptMPDB_Monomer_All.csv`  
- `CycPeptMPDB_Peptide_All.csv`  
- `monomer_table.csv`  
- `eval_index`  

Then, run the evaluation using:

``` 
cd ../../../; 
python -m experiments.0_benchmark.cpp.run_cpp --cache_path=experiments/embs/sincaa_cpp.pt  --exp_name=sincaa --model_type=${2} --num_layers=${1}  --pretrain_model_path=data/results/n1_weight0.1_innl2_both/ 
```
