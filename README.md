# SinCAA

**Learning representations for peptides containing non-canonical amino acids (ncAAs).**  
This repository provides the implementation, pretrained models, and training pipeline for the paper:

> **"Similarity-Enhanced Representation Learning of Non-Canonical Amino Acids for Therapeutic Peptide Modeling"**  
> *Chencheng Xu et al., 2025*

---

## 🚀 Overview
Standard protein language models are typically trained only on canonical amino acids, which limits their capacity to model peptides containing ncAAs.
This project introduces SinCAA, a framework that integrates a 3D conformational similarity metric into a graph transformer trained with dual objectives—contrastive learning and masked-node reconstruction—to generate transferable molecular embeddings capable of generalizing from individual ncAAs to complete peptides.
SinCAA demonstrates strong performance across multiple downstream tasks, including peptide binding affinity prediction, cell-penetrating ability estimation, and protein–peptide binding site prediction. Notably, it achieves substantial improvements over existing methods and exhibits remarkable zero-shot generalization, underscoring its potential to accelerate therapeutic peptide discovery.



## 🔥 Quick start
The pretrained weights for SinCAA are available at:
> data/results/n1_weight0.1_innl2_both/

To generate embeddings for peptides or amino acids from a CSV file containing molecular representations in the SMILES column, execute the following command:

> python -m Tools.generate_emb_from_smiles \
    --csv_path path_to_file.csv \
    --pretrained_dir data/results/n1_weight0.1_innl2_both/ \
    --save_dir save_path.npy


Here:
* --csv_path specifies the path to the input CSV file (e.g., data/examples/exam_csv.csv).

* --pretrained_dir provides the directory containing the pretrained SinCAA model weights.

* --save_dir defines the output path for saving the generated embeddings in NumPy format (.npy)

## 📂 Repository Structure

