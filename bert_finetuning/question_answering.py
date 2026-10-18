# %%
import collections
import os
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _bf16_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


# %% Config
@dataclass
class Config:
    model_name: str = "google-bert/bert-base-uncased"
    dataset_name: str = "rajpurkar/squad_v2"

    # Long contexts are split into overlapping windows of max_length tokens.
    max_length: int = 384
    doc_stride: int = 128  # overlap between consecutive windows

    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 2

    epochs: int = 2
    lr: float = 3e-5
    head_lr: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = field(default_factory=_bf16_available)

    # Answer decoding
    n_best: int = 20          # top-k start and end candidates per window
    max_answer_len: int = 30  # in tokens
    null_threshold: float = 0.0  # predict "no answer" unless best span beats null by this margin

    log_every: int = 200
    seed: int = 42
    output_dir: str = "checkpoints_qa"
    train_limit: Optional[int] = None  # e.g. 2000 for a quick smoke test
    eval_limit: Optional[int] = None


# %% Preprocessing
def prepare_train_features(examples: dict, tokenizer, cfg: Config) -> dict:
    """Tokenize (question, context) pairs into windows and label answer token spans.

    Unanswerable questions, and windows that don't contain the answer, are labelled
    with start = end = 0, i.e. the [CLS] token means "no answer".
    """
    enc = tokenizer(
        [q.lstrip() for q in examples["question"]],
        examples["context"],
        truncation="only_second",  # only ever truncate the context, never the question
        max_length=cfg.max_length,
        stride=cfg.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = enc.pop("overflow_to_sample_mapping")
    offset_mapping = enc.pop("offset_mapping")

    start_positions, end_positions = [], []
    for i, offsets in enumerate(offset_mapping):
        answers = examples["answers"][sample_map[i]]
        if len(answers["answer_start"]) == 0:  # unanswerable
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        seq_ids = enc.sequence_ids(i)  # None = special/pad, 0 = question, 1 = context
        ctx_start = seq_ids.index(1)
        ctx_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)

        # Answer not fully inside this window -> label as "no answer" for this window.
        if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
            continue

        s = ctx_start
        while s <= ctx_end and offsets[s][0] <= start_char:
            s += 1
        e = ctx_end
        while e >= ctx_start and offsets[e][1] >= end_char:
            e -= 1
        start_positions.append(s - 1)
        end_positions.append(e + 1)

    enc["start_positions"] = start_positions
    enc["end_positions"] = end_positions
    return enc


def prepare_eval_features(examples: dict, tokenizer, cfg: Config) -> dict:
    """Same windowing as training, but keep what's needed to map logits back to text."""
    enc = tokenizer(
        [q.lstrip() for q in examples["question"]],
        examples["context"],
        truncation="only_second",
        max_length=cfg.max_length,
        stride=cfg.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = enc.pop("overflow_to_sample_mapping")
    enc["example_id"] = []
    for i in range(len(enc["input_ids"])):
        seq_ids = enc.sequence_ids(i)
        enc["example_id"].append(examples["id"][sample_map[i]])
        # Null out non-context offsets so decoding can never pick a question/pad token.
        enc["offset_mapping"][i] = [
            off if seq_ids[k] == 1 else None
            for k, off in enumerate(enc["offset_mapping"][i])
        ]
    return enc


MODEL_INPUTS = ("input_ids", "attention_mask", "token_type_ids")
LABELS = ("start_positions", "end_positions")


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    # Features are already padded to max_length; just stack the tensor columns.
    # (offset_mapping / example_id stay on the dataset for post-processing.)
    keys = [k for k in (*MODEL_INPUTS, *LABELS) if k in batch[0]]
    return {k: torch.tensor([ex[k] for ex in batch], dtype=torch.long) for k in keys}


def build_data(cfg: Config, tokenizer):
    raw = load_dataset(cfg.dataset_name)
    train_examples, val_examples = raw["train"], raw["validation"]
    if cfg.train_limit:
        train_examples = train_examples.select(range(cfg.train_limit))
    if cfg.eval_limit:
        val_examples = val_examples.select(range(cfg.eval_limit))

    train_features = train_examples.map(
        prepare_train_features,
        fn_kwargs=dict(tokenizer=tokenizer, cfg=cfg),
        batched=True,
        remove_columns=train_examples.column_names,
    )
    val_features = val_examples.map(
        prepare_eval_features,
        fn_kwargs=dict(tokenizer=tokenizer, cfg=cfg),
        batched=True,
        remove_columns=val_examples.column_names,
    )
    print(
        f"train: {len(train_examples):,} examples -> {len(train_features):,} features | "
        f"val: {len(val_examples):,} examples -> {len(val_features):,} features"
    )

    common = dict(
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(
        train_features, batch_size=cfg.batch_size, shuffle=True, **common
    )
    # Must not shuffle: logits are matched back to val_features by position.
    val_loader = DataLoader(
        val_features, batch_size=cfg.eval_batch_size, shuffle=False, **common
    )
    return train_loader, val_loader, val_examples, val_features


# %% Model
class ExtractiveQAModel(nn.Module):
    """Encoder + a linear layer giving a start and an end logit for every token."""

    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.qa_head = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = {}
        if token_type_ids is not None:  # BERT uses segment ids; RoBERTa doesn't have them
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        ).last_hidden_state                       # (B, T, H)
        start_logits, end_logits = self.qa_head(hidden).unbind(dim=-1)  # (B, T) each
        return start_logits, end_logits


def make_optimizer(model: ExtractiveQAModel, cfg: Config):
    no_decay = ("bias", "LayerNorm.weight", "layernorm.weight", "norm.weight")

    def groups(module: nn.Module, lr: float) -> list[dict]:
        decay, nodecay = [], []
        for name, p in module.named_parameters():
            (nodecay if any(nd in name for nd in no_decay) else decay).append(p)
        return [
            {"params": decay, "lr": lr, "weight_decay": cfg.weight_decay},
            {"params": nodecay, "lr": lr, "weight_decay": 0.0},
        ]

    return torch.optim.AdamW(groups(model.encoder, cfg.lr) + groups(model.qa_head, cfg.head_lr))


# %% Post-processing: logits -> answer strings
def postprocess_predictions(
    examples,
    features,
    all_start_logits: np.ndarray,
    all_end_logits: np.ndarray,
    cfg: Config,
) -> dict[str, str]:
    feature_ids_per_example = collections.defaultdict(list)
    for i, ex_id in enumerate(features["example_id"]):
        feature_ids_per_example[ex_id].append(i)
    all_offsets = features["offset_mapping"]

    predictions = {}
    for ex in examples:
        context = ex["context"]
        best_score, best_text = -np.inf, ""
        min_null_score = np.inf

        for fi in feature_ids_per_example[ex["id"]]:
            start_logits, end_logits = all_start_logits[fi], all_end_logits[fi]
            offsets = all_offsets[fi]

            # Score of the "no answer" span ([CLS], [CLS]) in this window.
            min_null_score = min(min_null_score, start_logits[0] + end_logits[0])

            start_idx = np.argsort(start_logits)[-1 : -cfg.n_best - 1 : -1]
            end_idx = np.argsort(end_logits)[-1 : -cfg.n_best - 1 : -1]
            for s in start_idx:
                for e in end_idx:
                    if offsets[s] is None or offsets[e] is None:
                        continue  # not in the context
                    if e < s or e - s + 1 > cfg.max_answer_len:
                        continue
                    score = start_logits[s] + end_logits[e]
                    if score > best_score:
                        best_score = score
                        best_text = context[offsets[s][0] : offsets[e][1]]

        has_answer = best_score > min_null_score + cfg.null_threshold
        predictions[ex["id"]] = best_text if has_answer else ""
    return predictions


# %% SQuAD v2 metrics (same normalisation as the official script)
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    pred_toks, gold_toks = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pred_toks or not gold_toks:  # no-answer case: both must be empty
        return float(pred_toks == gold_toks)
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision, recall = n_same / len(pred_toks), n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def squad_v2_metrics(examples, predictions: dict[str, str]) -> dict[str, float]:
    scores = {"has_ans": ([], []), "no_ans": ([], [])}
    for ex in examples:
        golds = ex["answers"]["text"] or [""]  # unanswerable -> gold is the empty string
        pred = predictions[ex["id"]]
        em = max(float(normalize_answer(pred) == normalize_answer(g)) for g in golds)
        f1 = max(_f1(pred, g) for g in golds)
        bucket = "has_ans" if ex["answers"]["text"] else "no_ans"
        scores[bucket][0].append(em)
        scores[bucket][1].append(f1)

    all_em = scores["has_ans"][0] + scores["no_ans"][0]
    all_f1 = scores["has_ans"][1] + scores["no_ans"][1]
    out = {"exact": 100 * np.mean(all_em), "f1": 100 * np.mean(all_f1)}
    for bucket, (ems, f1s) in scores.items():
        if ems:
            out[f"{bucket}_exact"] = 100 * np.mean(ems)
            out[f"{bucket}_f1"] = 100 * np.mean(f1s)
    return out


# %% Trainer
class QATrainer:
    def __init__(self, model, train_loader, val_loader, val_examples, val_features, device, cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_examples = val_examples
        self.val_features = val_features
        self.device = device
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_f1 = -1.0
        self.ckpt_path = Path(cfg.output_dir) / "best_model.pt"

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.cfg.bf16
        )

    def _inputs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device, non_blocking=True) for k in MODEL_INPUTS if k in batch}

    def train_one_epoch(self, optimizer, scheduler, epoch: int) -> float:
        self.model.train()
        running, total = 0.0, 0.0
        n_steps = len(self.train_loader)
        for step, batch in enumerate(self.train_loader, start=1):
            start_pos = batch["start_positions"].to(self.device)
            end_pos = batch["end_positions"].to(self.device)
            with self._autocast():
                start_logits, end_logits = self.model(**self._inputs(batch))
            # Loss in fp32: average of start and end cross-entropies.
            loss = (
                self.loss_fn(start_logits.float(), start_pos)
                + self.loss_fn(end_logits.float(), end_pos)
            ) / 2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            total += loss.item()
            if step % self.cfg.log_every == 0:
                print(
                    f"epoch {epoch} | step {step}/{n_steps} | "
                    f"loss {running / self.cfg.log_every:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
                )
                running = 0.0
        return total / n_steps

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        starts, ends = [], []
        for batch in self.val_loader:
            with self._autocast():
                s, e = self.model(**self._inputs(batch))
            starts.append(s.float().cpu().numpy())
            ends.append(e.float().cpu().numpy())
        preds = postprocess_predictions(
            self.val_examples, self.val_features,
            np.concatenate(starts), np.concatenate(ends), self.cfg,
        )
        return squad_v2_metrics(self.val_examples, preds)

    def fit(self) -> None:
        optimizer = make_optimizer(self.model, self.cfg)
        total_steps = self.cfg.epochs * len(self.train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.cfg.warmup_ratio * total_steps), total_steps
        )
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_one_epoch(optimizer, scheduler, epoch)
            m = self.evaluate()
            print(
                f"== epoch {epoch} | train loss {train_loss:.4f} | "
                f"EM {m['exact']:.2f} | F1 {m['f1']:.2f} | "
                f"HasAns F1 {m.get('has_ans_f1', float('nan')):.2f} | "
                f"NoAns F1 {m.get('no_ans_f1', float('nan')):.2f}"
            )
            if m["f1"] > self.best_f1:
                self.best_f1 = m["f1"]
                self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self.ckpt_path)
                print(f"   saved new best (F1 {m['f1']:.2f}) -> {self.ckpt_path}")


# %% Main
def main(cfg: Optional[Config] = None) -> ExtractiveQAModel:
    cfg = cfg or Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | bf16={cfg.bf16}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    assert tokenizer.is_fast, "offset mappings need a fast tokenizer"
    train_loader, val_loader, val_examples, val_features = build_data(cfg, tokenizer)

    model = ExtractiveQAModel(cfg.model_name).to(device)
    trainer = QATrainer(model, train_loader, val_loader, val_examples, val_features, device, cfg)
    trainer.fit()
    print(f"\nbest val F1: {trainer.best_f1:.2f}")
    return model


if __name__ == "__main__":
    main()