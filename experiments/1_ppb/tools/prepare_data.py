from arguments import parser
from utils.prepare_data import single_data_preprocess
from tqdm import tqdm
parser.add_argument("--line", type=str)
parser.add_argument("--cutoff", type=float,nargs='*', default=[])
args=parser.parse_args()
args.emb_path="vocab_emb"
if len(args.cutoff)==0:
    update_protein_only=True
else:
    update_protein_only=False
single_data_preprocess(args.line, "raw_dir", args.data_dir, args, update_peptide_only=False, update_protein_only=update_protein_only,cutoffs=args.cutoff)