# -*- coding: utf-8 -*-
"""
@File   :  main.py
@Time   :  2024/02/08 14:40
@Author :  Yufan Liu
@Desc   :  Main function for training and testing
"""


import os
import time

import pytorch_lightning as pl
import yaml
from pytorch_lightning import Trainer

from arguments import parser
from run.dataset import SurfBindDataModule
from run.trainer import SurfBindTrainer, early_stop, trainer_checkpoint
from utils.utilities import cpu_monitor, print_model_parameters
from pytorch_lightning.loggers import CSVLogger


time_tic = time.strftime("%m-%d-%H-%M", time.localtime())
args = parser.parse_args()
print("Total memory available:", cpu_monitor("total"))
pl.seed_everything(args.random_seed)

# training settings
clu_type = args.processed_dir.split("/")[-2]
if args.extra_emb:
    save_folder = f"experiments_higmae_v0811/{clu_type}/{args.exper_setting}/{args.fold}"
else:
    save_folder = f"experiments_v0811/{clu_type}/{args.exper_setting}/{args.fold}"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
else:
    print("already exists", save_folder)
    exit()

with open(f"{save_folder}/args.yaml", "w") as file:
    yaml.dump(vars(args), file)

checkpoint = trainer_checkpoint(save_folder)
# early_stopping = early_stop()
csv_logger = CSVLogger(
    save_dir=save_folder,      # 保存日志的主目录
    name="log.csv",  # 实验名称，最终路径为 logs/surfbind_logs/version_*
)
# data, model, and train
data_module = SurfBindDataModule(
    data_dir=args.data_dir,
    processed_dir=args.processed_dir,
    exper_setting=args.exper_setting,
    fold=args.fold,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    seed=args.random_seed,
    threshold=args.dist_threshold,
    extra_emb=args.extra_emb
)
args.save_folder = save_folder
model = SurfBindTrainer(args)
exist_model_path = os.path.join(save_folder, "best.ckpt")
if os.path.exists(exist_model_path):
    model = SurfBindTrainer.load_from_checkpoint(
        exist_model_path, **{"args": args})
    print("loading from exists", exist_model_path)
print_model_parameters(model)
trainer = Trainer(
    max_epochs=args.epochs,
    logger=csv_logger,
    callbacks=[checkpoint],
    log_every_n_steps=args.log_steps,
    check_val_every_n_epoch=args.test_epochs,
    accelerator="gpu"
)

trainer.fit(model, datamodule=data_module)

# final test
best_model_path = checkpoint.best_model_path
model = SurfBindTrainer.load_from_checkpoint(best_model_path, **{"args": args})
trainer.test(model, datamodule=data_module)
