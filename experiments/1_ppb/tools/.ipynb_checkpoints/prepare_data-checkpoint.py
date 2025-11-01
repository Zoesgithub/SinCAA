from arguments import parser
from utils.prepare_data import single_data_preprocess
from tqdm import tqdm
parser.add_argument("--line", type=str)
parser.add_argument("--cutoff", type=float,nargs='*', default=[])
args=parser.parse_args()
args.emb_path="vocab_emb"

single_data_preprocess(args.line, "raw_dir", args.data_dir, args, update_peptide_only=False, cutoffs=args.cutoff)