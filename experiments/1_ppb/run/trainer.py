# -*- coding: utf-8 -*-

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import roc_auc_score, average_precision_score

from run.models import FullModel
import pickle


class SurfBindTrainer(pl.LightningModule):
    """A trainer class, for training and testing"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.epochs = args.epochs
        self.model = FullModel(args).to(args.device)
        self.train_output = [[], []]
        self.validation_output = [[], []]
        self.test_output = [[], []]
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.best_val_loss = float("inf")

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.train_output = [[], []]

    def training_step(self, batch, batch_idx):
        self.train()
        outputs = self(batch)
        loss, score = self.compute_loss(outputs, batch)
        self.train_output[0].append(loss)
        self.train_output[1].extend(score)
        return loss

    def on_train_epoch_end(self):
        loss, auprc = self.metric_on_epoch(self.train_output)
        self.train_output = [[], []]
        self.log("train_epoch_loss", loss)
        self.log("train_pro_auprc", auprc[0])
        self.log("train_pep_auprc", auprc[1])
        self.log("train_pro_auc", auprc[2])
        self.log("train_pep_auc", auprc[3])

    def on_validation_start(self):
        self.validation_output = [[], []]

    def validation_step(self, batch, batch_idx):
        self.eval()
        outputs = self(batch)
        loss, score = self.compute_loss(outputs, batch)
        self.validation_output[0].append(loss)
        self.validation_output[1].extend(score)
        return loss

    def on_validation_epoch_end(self):
        loss, auprc = self.metric_on_epoch(self.validation_output)
        self.validation_output = [[], []]
        self.log("valid_epoch_loss", loss)
        self.log("valid_pro_auprc", auprc[0])
        self.log("valid_pep_auprc", auprc[1])
        self.log("valid_pro_auc", auprc[2])
        self.log("valid_pep_auc", auprc[3])

    def on_test_start(self):
        self.test_output = [[], []]

    def test_step(self, batch, batch_idx):
        self.eval()
        outputs = self(batch)
        loss, score = self.compute_loss(outputs, batch)
        self.test_output[0].append(loss)
        self.test_output[1].extend(score)
        return loss

    def on_test_epoch_end(self):
        loss, auprc = self.metric_on_epoch(self.test_output)

        self.log("test_epoch_loss", loss)
        self.log("test_pro_auprc", auprc[0])
        self.log("test_pep_auprc", auprc[1])
        self.log("test_pro_auc", auprc[2])
        self.log("test_pep_auc", auprc[3])
        save_path = f"{self.args.save_folder}/testres.pickle"
        with open(save_path, "wb") as f:
            pickle.dump(self.test_output[1], f)
        self.test_output = [[], []]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate)
        return optimizer

    def compute_loss(self, outputs, batch, pos_weight=50):
        pred, tpro_pred, tpep_pred = outputs
        label = batch[-2]
        pep_residue_index = batch[0].residue_index
        if hasattr(batch[0], "label_residue_index"):
            label_residue_index = batch[0].label_residue_index
        else:
            label_residue_index = batch[0].residue_index
        assert len(pred) == len(label), f"{len(pred)} {len(label)}"
        assert len(pred) == len(pep_residue_index)
        loss = []
        pro_label = []
        pep_label = []
        pro_pred = []
        pep_pred = []
        ret = []
        for p, pp, tpep, l, r, l_r in zip(pred, tpro_pred, tpep_pred, label, pep_residue_index, label_residue_index):
            l = torch.tensor(l).to(p.device)
            r = torch.tensor(r).to(p.device)
            l_r = torch.tensor(l_r).to(p.device)

            pro_l = l.max(0).values  # l is peptide x protein
            pep_l = l.max(1).values
            pep_l = torch.scatter_reduce(pep_l.new_zeros(
                l_r.max()+1), 0,  l_r, pep_l, include_self=False, reduce="amax")
            l = torch.scatter_reduce(l.new_zeros(l_r.max(
            )+1, l.shape[-1]), 0,  l_r[..., None].expand_as(l), l, include_self=False, reduce="amax")
            m = l > -1
            pro_p = pp
            pep_p = tpep
            assert pro_p.shape == pro_l.shape, f"{ pro_p.shape} {pro_l.shape}"
            assert pep_p.shape == pep_l.shape
            tp = p  # *pro_p[None]*tpep[:, None]
            loss.append(
                -(pos_weight*l*torch.log(tp.clamp(1e-8))+(1-l)
                  * torch.log((1-tp).clamp(1e-8)))[m].mean()
                - (10*pro_l*torch.log(pro_p.clamp(1e-8))+(1-pro_l)
                   * torch.log((1-pro_p).clamp(1e-8)))[pro_l > -1].mean()
                - (pep_l*torch.log(pep_p.clamp(1e-8))+(1-pep_l) *
                   torch.log((1-pep_p).clamp(1e-8)))[pep_l > -1].mean()
            )

            assert pep_p.shape[0] == l_r.max()+1, pep_p.shape
            pro_label.append(pro_l)
            pep_label.append(pep_l)
            pro_pred.append(pro_p)
            pep_pred.append(pep_p)
            ret.append({"pro_pred": pro_p, "pro_l": pro_l,
                       "pep_l": pep_l, "pep_pred": pep_p, "p_pred": p, "p_l": l})
        loss = sum(loss)/len(pred)

        # (torch.cat(pro_pred,0).detach(),torch.cat(pro_label,0).detach(),torch.cat( pep_pred,0).detach(), torch.cat( pep_label,0).detach())  # score
        return loss, ret

    def metric_on_epoch(self, outputs):
        """In this case, only loss and scores are input and return average values."""
        loss, score = outputs

        mean_loss = sum(loss)/len(loss)

        pro_pred = torch.cat([_["pro_pred"] for _ in score])
        pro_label = torch.cat([_["pro_l"] for _ in score])

        pep_pred = torch.cat([_["pep_pred"] for _ in score])
        pep_label = torch.cat([_["pep_l"] for _ in score])

        pro_auprc = average_precision_score(
            pro_label[pro_label > -1].detach().cpu().numpy(), pro_pred[pro_label > -1].detach().cpu().numpy())
        pep_auprc = average_precision_score(
            pep_label[pep_label > -1].detach().cpu().numpy(), pep_pred[pep_label > -1].detach().cpu().numpy())
        pro_auc = roc_auc_score(pro_label[pro_label > -1].detach().cpu(
        ).numpy(), pro_pred[pro_label > -1].detach().cpu().numpy())
        print(pep_label[pep_label > -1].sum(), pep_label[pep_label > -1].shape)
        try:
            pep_auc = roc_auc_score(pep_label[pep_label > -1].detach().cpu(
            ).numpy(), pep_pred[pep_label > -1].detach().cpu().numpy())
        except:
            pep_auc = -1

        return mean_loss, (pro_auprc, pep_auprc, pro_auc, pep_auc)


def trainer_checkpoint(save_folder):
    return ModelCheckpoint(
        dirpath=save_folder,
        filename="best",
        save_top_k=1,
        monitor="valid_pro_auprc",
        mode="max",
        save_weights_only=True,
    )


def early_stop():
    # if valid auc not improved for 4 times (4 * test_epochs, trainng stop)
    return EarlyStopping(monitor="train_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
