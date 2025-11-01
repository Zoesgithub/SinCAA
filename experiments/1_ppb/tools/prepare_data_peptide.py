from arguments import parser
from utils.prepare_data import single_data_preprocess
from tqdm import tqdm
from rdkit import RDLogger
import sys
import os
prefix="/".join(os.getcwd().split("/")[:-2]) 
sys.path.append(prefix)
from experiments.utils import load_model, get_emb_from_feat
sys.path.pop(-1)
pretrained_sincaa=load_model(os.path.join(prefix, "data/results/n1_weight0.1_innl2_both/")).cuda()
RDLogger.DisableLog('rdApp.*')

parser.add_argument("--file", type=str)
parser.add_argument("--cutoff", type=float,nargs='*', default=[])
parser.add_argument("--start", type=int, default=0)
args=parser.parse_args()
args.emb_path="vocab_emb"
steps=500
with open(args.file, "r") as f:
    lines=f.readlines()
for l in tqdm(lines[args.start:]):
    try:
        single_data_preprocess(l.replace("\n", ""), "raw_dir", args.data_dir, args, update_peptide_only=True, cutoffs=args.cutoff, pretrained_sincaa=[pretrained_sincaa,get_emb_from_feat])
    except Exception as e:
        print(e, l)
        continue