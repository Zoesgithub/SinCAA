from argparse import ArgumentParser
import json
import os
from loguru import logger
from utils.train_utils import trainer

def main(args):
    trainer(args)

if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of experiment")
    parser.add_argument("--save_path", type=str, help="Path to save results", default="data/results/")
    
    parser.add_argument("--train_aa_data_path", type=str, help="the path to the train aa data", default="data/AAList/train_aa_with_similarity.csv")
    parser.add_argument("--train_mol_data_path", type=str, help="the path to train general mol", default="data/ZINC15/train_zinc15_10M_2D.csv")
    
    parser.add_argument("--val_aa_data_path", type=str, help="the path to the val aa data", default="data/AAList/val_aa_with_similarity.csv")
    parser.add_argument("--val_mol_data_path", type=str, help="the path to val general mol", default="data/ZINC15/val_zinc15_10M_2D.csv")
    
    parser.add_argument("--model_channels", type=int, default=256, help="The number of channels for model")
    parser.add_argument("--num_head", type=int, default=16, help="The number of channels for model")
    parser.add_argument("--topological_net_layers", type=int, default=6, help="The number of topological net layers")
    parser.add_argument("--decoder_layers", type=int, default=2, help="The number of decoder layers")
    
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10000, help="The number of training epochs")
    #parser.add_argument("--validate_data_size", type=int, default=2, help="The number of data points for validation")
    parser.add_argument("--logger_step", type=int, help="Step for logger", default=5)
    parser.add_argument("--cache_path", type=str, help="path to cache files", default="data/cache/")
    parser.add_argument("--load_path", type=str, help="path to load state dict", default=None)
    parser.add_argument("--num_workers", type=int, help="num of workers to load data", default=5)
    parser.add_argument("--max_combine", type=int, help="", default=4)
    parser.add_argument("--norm", type=str, choices=["BatchNorm", "GraphNorm", "LayerNorm", "None"], default=None)
    parser.add_argument("--aba", action='store_true')
    parser.add_argument("--model",  type=str, choices=["GPS"], default="GPS")
    
    args=parser.parse_args()
    if args.norm=="None":
        args.norm=None
    save_path=os.path.join(args.save_path, args.experiment_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(os.path.join(save_path, "config.json"), "w") as f:
        f.write(json.dumps(vars(args)))
    logger.add(os.path.join(save_path, "log"))
    main(args)