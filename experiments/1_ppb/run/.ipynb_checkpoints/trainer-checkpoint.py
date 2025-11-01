# -*- coding: utf-8 -*-

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import roc_auc_score

from run.model import SurfaceBind



class SurfBindTrainer(pl.LightningModule):
    """A trainer class, for training and testing"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.epochs = args.epochs
        self.model = SurfaceBind(args).to(args.device)
        self.train_output = []
        self.validation_output = []
        self.test_output = []
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.train_output = []

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss, score = self.compute_loss(outputs)
        self.train_output.append((loss, score))
        self.log("train_step_loss", loss)
        self.log("positive_mean", score[0].mean())
        self.log("negative_mean", score[1].mean())
        return loss

    def on_train_epoch_end(self):
        loss, auc = self.metric_on_epoch(self.train_output)
        self.train_output.clear()
        self.log("train_used_cpu", cpu_monitor())
        self.log("train_epoch_loss", loss)
        self.log("train_auc", auc)

    def on_validation_start(self):
        self.validation_output = []

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss, score = self.compute_loss(outputs)
        self.validation_output.append((loss, score))
        return loss

    def on_validation_epoch_end(self):
        loss, auc = self.metric_on_epoch(self.validation_output)
        self.validation_output.clear()
        self.log("valid_used_cpu", cpu_monitor())
        self.log("valid_epoch_loss", loss)
        self.log("valid_auc", auc)

    def on_test_start(self):
        self.test_output = []

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss, score = self.compute_loss(outputs)
        self.test_output.append((loss, score))
        return loss

    def on_test_epoch_end(self):
        loss, auc = self.metric_on_epoch(self.test_output)
        self.test_output.clear()
        self.log("test_used_cpu", cpu_monitor())
        self.log("test_epoch_loss", loss)
        self.log("test_auc", auc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return optimizer

    def compute_roc_auc(self, pos, neg):
        pos = pos.detach().cpu().numpy()
        neg = neg.detach().cpu().numpy()
        labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
        dist_pairs = np.concatenate([pos, neg])

        return roc_auc_score(labels, dist_pairs)

    def compute_loss(self, outputs):
        dist_p, dist_n, pep_p, pep_n = outputs
        loss_pair = -torch.log(1 - dist_p + 1e-8).mean() - torch.log(dist_n + 1e-8).mean()
        # pep_target = torch.tensor([1] * pep_p.shape[0] + [0] * pep_n.shape[0]).to(self.args.device)
        # loss_pep = self.ce_loss(torch.cat([pep_p, pep_n], dim=0), pep_target)

        loss = loss_pair
        return loss, (dist_p, dist_n)  # score

    def metric_on_epoch(self, outputs):
        """In this case, only loss and scores are input and return average values."""
        loss = [x[0] for x in outputs]
        score = [x[1] for x in outputs]

        mean_loss = torch.stack(loss).mean()
        pos = torch.cat([d[0] for d in score])
        neg = torch.cat([d[1] for d in score])
        auc = 1 - self.compute_roc_auc(pos, neg)
        return mean_loss, auc

def trainer_checkpoint(save_folder):
    return ModelCheckpoint(
        dirpath=save_folder,
        filename="{epoch}-{step}-{train_auc:.4f}",
        save_top_k=2,
        monitor="valid_auc",
        mode="max",
        save_weights_only=True,
    )


def early_stop():
    # if valid auc not improved for 4 times (4 * test_epochs, trainng stop)
    return EarlyStopping(monitor="train_auc", min_delta=0.00, patience=10, verbose=False, mode="max")
