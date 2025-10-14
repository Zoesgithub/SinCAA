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



## 📂 Quick start

## 📂 Repository Structure

