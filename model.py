import torch
from torch import nn
from transformers import AutoModel
from dataset import CommonDataSet, TestDataSet


class RumourCLS(nn.Module):
    def __init__(self, pre_encoder):

        super(RumourCLS, self).__init__()
        self.encoder = AutoModel.from_pretrained(pre_encoder)
        hidden_size = self.encoder.config.hidden_size
        self.cls = nn.Linear(hidden_size, 2)
        # self.gru = nn.GRUCell(hidden_size, hidden_size)
        # self.evidence_cls = nn.Linear(hidden_size * 2, 2)

    def forward(self, reps, masks):
        texts_emb = self.encoder(input_ids=reps, attention_mask=masks).last_hidden_state
        # first token
        texts_emb = texts_emb[:, 0, :]
        logits = self.cls(texts_emb)

        # gru
        # hiddens = []
        # input_ids = reps.tolist()
        # for i in range(reps.size(0)):
        #     hidden = texts_emb[i, 0, :].view(1, -1)
        #     for jidx, j in enumerate(input_ids[0]):
        #         if j == 2:
        #             hidden = self.gru(texts_emb[i, jidx, :].view(1, -1), hidden)
        #     hiddens.append(hidden)
        # logits = self.cls(torch.cat(hiddens, 0))

        # evidence
        # logits = []
        # input_ids = reps.tolist()
        # for i in range(reps.size(0)):
        #     sent_sep_index = []
        #     for jidx, j in enumerate(input_ids[0]):
        #         if j == 2:
        #             sent_sep_index.append(jidx)
        #     sent_rep = torch.index_select(texts_emb[i, :, :], 0, torch.Tensor(sent_sep_index).type_as(reps).long())
        #     first_sent = sent_rep[0].view(1, -1).repeat(sent_rep.size(0), 1)
        #     logit = self.evidence_cls(torch.cat([first_sent, sent_rep], 1))
        #     logits.append(torch.mean(logit, 0))
        # logits = torch.stack(logits, 0)

        return logits
#!/usr/bin/env python
import argparse
from sklearn.metrics import precision_recall_fscore_support

import torch
# from models import RumourCLS
from pathlib import Path
from transformers import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import AdamW
from pytorch_lightning import LightningModule

from torch.utils.data import DataLoader


class RumourDetection(LightningModule):
    def __init__(self, hparams):
        # save hparams as a Namespace if hparams is a dict
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        """Initialize model and tokenizer."""
        super().__init__()
        self.save_hyperparameters(hparams)
        # self.hparams = hparams

        self.tokenizer = AutoTokenizer.from_pretrained(hparams.pre_encoder)
        self.cls = RumourCLS(hparams.pre_encoder)

        self.output_dir = Path(hparams.output_dir)
        self.metrics_save_path = Path(hparams.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        self.metrics = dict()
        self.metrics["val"] = []
        self.val_metric = "f1_score"

        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

    def re_init(self, hparams):
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters(hparams)
        # self.hparams = hparams

        self.tokenizer = AutoTokenizer.from_pretrained(hparams.pre_encoder)

        self.output_dir = Path(hparams.output_dir)
        self.metrics_save_path = Path(hparams.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        self.metrics = dict()
        self.metrics["val"] = []
        self.val_metric = "f1_score"

        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches
        return (len(self.train_dataloader().dataset) / effective_batch_size) * self.hparams.max_epochs

    def configure_optimizers(self):
        """Prepare Adafactor optimizer and schedule"""
        no_decay = ['bias', 'LayerNorm.weight']
        encoder_parameters = [
            {'params': [p for n, p in self.cls.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.cls.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        encoder_optimizer = AdamW(encoder_parameters, lr=self.hparams.learning_rate)

        total_steps = int(self.total_steps())
        encoder_scheduler = get_cosine_schedule_with_warmup(encoder_optimizer, num_warmup_steps=int(self.hparams.warmup_ratio * total_steps), num_training_steps=total_steps)
        encoder_scheduler = {"scheduler": encoder_scheduler, "interval": "step", "frequency": 1}
        return [encoder_optimizer], [encoder_scheduler]

    def training_step(self, batch, batch_idx):
        logits = self.cls(batch["input_text"], batch["attn_text"])
        labels = batch["label"]
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if batch_idx % self.hparams.log_every_n_steps == 0:
            self.logger.log_metrics({"loss": loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.cls(batch["input_text"], batch["attn_text"])
        preds = torch.argmax(logits, dim=1).tolist()
        labels = batch["label"]
        return {"preds": preds, "labels": labels.tolist()}

    def test_step(self, batch, batch_idx):
        logits = self.cls(batch["input_text"], batch["attn_text"])
        preds = torch.argmax(logits, dim=1).tolist()
        return {"preds": preds}
        # return {"preds": preds, "origin_tweet": batch["origin_tweet"]}

    def get_dataloader(self, type_path, batch_size, shuffle=False):
        n_obs = self.n_obs[type_path]
        if type_path == "train":
            dataset = CommonDataSet(
                tokenizer=self.tokenizer,
                max_length=self.hparams.max_length,
                data_dir="./data/crawl-data/train/",
                data_file_name="./data/project-data/train.data.txt",
                label_file_name="./data/project-data/train.label.txt",
                n_obs=n_obs,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.hparams.num_workers,
                sampler=None,
                drop_last=True,
            )
        elif type_path == "val":
            dataset = CommonDataSet(
                tokenizer=self.tokenizer,
                max_length=self.hparams.max_length,
                data_dir="./data/crawl-data/dev/",
                data_file_name="./data/project-data/dev.data.txt",
                label_file_name="./data/project-data/dev.label.txt",
                n_obs=n_obs
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.hparams.num_workers,
                sampler=None,
            )
        else:
            dataset = TestDataSet(
                tokenizer=self.tokenizer,
                max_length=self.hparams.max_length,
                data_dir="./data/crawl-data/test/",
                data_file_name="./data/project-data/test.data.txt",
                n_obs=n_obs,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.hparams.num_workers,
                sampler=None,
            )

    def train_dataloader(self):
        return self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self):
        return self.get_dataloader("test", batch_size=self.hparams.test_batch_size)

    def validation_epoch_end(self, outputs):
        preds = []
        labels = []
        for x in outputs:
            for pred, label in zip(x["preds"], x["labels"]):
                preds.append(pred)
                labels.append(label)

        p, r, f, _ = precision_recall_fscore_support(labels, preds, pos_label=1, average="binary")
        self.log(self.val_metric, f, logger=False)

        val_metrics = dict()
        val_metrics[f"val_precision"] = p
        val_metrics[f"val_recall"] = r
        val_metrics[f"val_f1_score"] = f
        self.logger.log_metrics(val_metrics)
        self.metrics["val"].append(val_metrics)

    def test_epoch_end(self, outputs):
        preds = []
        for x in outputs:
            for pred in x["preds"]:
                preds.append(pred)

        # Log results
        od = Path(self.hparams.output_dir)
        # results_file = od / "test.predictions.csv"
        results_file = od / "submissions.csv"

        with open(results_file, "w") as writer:
            writer.write("Id,Predicted\n")
            for xid, x in enumerate(preds):
                writer.write(str(xid) + "," + str(x) + "\n")

        # preds = []
        # tweets = []
        # for x in outputs:
        #     for pred, tweet in zip(x["preds"], x["origin_tweet"]):
        #         preds.append(pred)
        #         tweets.append(tweet)
        #
        # od = Path(self.hparams.output_dir)
        # rumour_file = od / "rumour.jsonl"
        # nonrumour_file = od / "nonrumour.jsonl"
        #
        # rumour_writer = open(rumour_file, "w")
        # nonrumour_writer = open(nonrumour_file, "w")
        # for x, y in zip(preds, tweets):
        #     if x == 0:
        #         nonrumour_writer.write(y + "\n")
        #     else:
        #         rumour_writer.write(y + "\n")
