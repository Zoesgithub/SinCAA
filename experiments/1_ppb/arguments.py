# -*- coding: utf-8 -*-
"""
@File   :  arguments.py
@Time   :  2024/02/05 18:52
@Author :  Yufan Liu
@Desc   :  args
"""

import argparse

parser = argparse.ArgumentParser()
'''parser.add_argument(
    "--REDUCE_BIN", type=str, default="tools/reduce/reduce_src/reduce"
)  # prebuild by yourself
parser.add_argument("--MSMS_BIN", type=str, default="tools/msms/msms.x86_64Linux2.2.6.1")
'''

# general parameters
parser.add_argument("--random_seed", type=int, default=0, help="seed.")

# protein surface parameters
'''parser.add_argument(
    "--collapse_rate",
    type=float,
    default=0.2,
    help="Collapse rate for surface optimization",
)
parser.add_argument("--max_vertex", type=int, default=80, help="Max vertex neighbor for input feature.")
parser.add_argument("--sc_radius", type=int, default=5, help="patch radius")
parser.add_argument("--max_distance", type=float, default=5.0, help="Max patch radius for searching")
'''

# model training settings
parser.add_argument("--n_layers", type=int, default=6)
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--n_head", type=int, default=6)
parser.add_argument("--d_k", type=int, default=64)
parser.add_argument("--d_v", type=int, default=128)
parser.add_argument("--d_inner", type=int, default=64)

parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--test_epochs", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda",
                    help="device for model")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Batch size for training")
parser.add_argument("--num_workers", type=int, default=2,
                    help="Number of workers for data loader")
parser.add_argument("--log_steps", type=int, default=1,
                    help="Log every n steps in pl trainer")

# loss computation
# parser.add_argument("--pos_thresh", type=float, default=0.0, help="Threshold for positive samples")
# parser.add_argument("--neg_thresh", type=float, default=10.0, help="Threshold for negative samples")
# parser.add_argument("--vocab_length", type=int, default=8, help="The length of vocabulary of atom type")

# initialization MaSIF model
'''
parser.add_argument("--n_thetas", type=int, default=16, help="Number of thetas for grid generation")
parser.add_argument("--n_rhos", type=int, default=5, help="Number of rhos for grid generation")
parser.add_argument("--n_rotations", type=int, default=8, help="Number of rotations for feature max-out")
parser.add_argument("--n_features", type=int, default=5, help="Number of features")
'''
parser.add_argument(
    "--dist_threshold",
    type=float,
    choices=[0.657, 0.956, 1.543, 1.983, 2.443],
    help="threshold for data",
)


# inference part
parser.add_argument(
    "--inference_file",
    type=str,
    default=None,
    help="a txt file as data format for inference",
)
'''parser.add_argument(
    "--vertex_n",
    type=int,
    default=1,
    help="positive binding peptide residue if exceeded this vertex_num",
)'''
parser.add_argument("--peptide_name", type=str, default=None,
                    help="peptide name for interwine prediction")
parser.add_argument("--protein_name", type=str, default=None,
                    help="protein name for interwine prediction")


parser.add_argument("--cache_model", type=str, default=None,
                    help="model caching used for inference")
parser.add_argument(
    "--mode",
    type=str,
    default=None,
    help="used as an arbitary value do whatever u want.",
)
# this is set as an arbitary value for versatile utilities,
# check the scripts for details, like protein id in preprocessing
# like inference data list


# aux settings
parser.add_argument(
    "--data_dir",
    type=str,
    default="data/peptide_data/processed_cv_pepnn",
    help="path for preprocessed data",
)
parser.add_argument(
    "--extra_emb",
    type=bool,
    default=False
)
# lyf: actually the name here is a bit confused, the data_dir is actually the processed feature
# labels etc., and the processed_dir is the file lists of names like "1A0N_A_B"
# but fix this is not necessary.
parser.add_argument(
    "--exper_setting",
    type=str,
    default="novel_protein",
    help="splitting setting for data",
)
parser.add_argument(
    "--processed_dir",
    type=str,
    default="data/peptide_data/data_source/",
    help="storage of processed files",
)
parser.add_argument("--fold", type=str, default="fold_0",
                    help="for cross validation")
