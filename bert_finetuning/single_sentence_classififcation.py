# %%
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

# Tokenization happens inside DataLoader workers; silence the HF fork warning.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# %% Config
@dataclass
class Config:
    model_name: str = "google-bert/bert-base-uncased"
    task: str = "sst2"
    max_length: int = 128
    batch_size: int = 32
    num_workers: int = 2

    # Classifier head
    hidden_dims: tuple[int, int] = (512, 256)
    dropout: float = 0.1

    # Phase 1: backbone frozen, train the head only
    frozen_epochs: int = 1
    frozen_head_lr: float = 5e-4

    # Phase 2: backbone unfrozen, train BERT + head together
    finetune_epochs: int = 2
    backbone_lr: float = 2e-5
    head_lr: float = 1e-4
    warmup_ratio: float = 0.1

    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    log_every: int = 200
    seed: int = 42
    output_dir: str = "checkpoints"


# %% Model
class SentimentClassificationBert(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        hidden_dims: tuple[int, int] = (512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)

        h1, h2 = hidden_dims
        # 3 linear layers: 768 -> 512 -> 256 -> num_labels
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, num_labels),  # no activation: raw logits
        )
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.backbone_frozen = True
        self.bert.requires_grad_(False)
        self.bert.eval()

    def unfreeze_backbone(self) -> None:
        self.backbone_frozen = False
        self.bert.requires_grad_(True)
        self.bert.train(self.training)  # follow the parent's current mode

    def train(self, mode: bool = True):
        super().train(mode)
        # While frozen, BERT is a fixed feature extractor: keep its dropout off.
        if self.backbone_frozen:
            self.bert.eval()
        return self

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Don't build an autograd graph through BERT while it's frozen.
        # (`is_grad_enabled()` keeps an outer torch.no_grad(), e.g. in evaluate, in charge.)
        grad_through_bert = torch.is_grad_enabled() and not self.backbone_frozen
        with torch.set_grad_enabled(grad_through_bert):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        # Raw logits; CrossEntropyLoss applies log-softmax internally.
        return self.classifier(cls_embedding)


def count_params(model: nn.Module) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return f"trainable params: {trainable:,} / {total:,}"


# %% Data
class Collator:
    """Tokenize raw SST-2 examples and pad to the longest sentence in the batch."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            [ex["sentence"] for ex in batch],
            padding=True,  # dynamic padding: only as long as this batch needs
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor([ex["label"] for ex in batch], dtype=torch.long),
        }


def build_dataloaders(cfg: Config, tokenizer) -> tuple[DataLoader, DataLoader]:
    dataset = load_dataset("nyu-mll/glue", cfg.task)
    common = dict(
        collate_fn=Collator(tokenizer, cfg.max_length),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # GLUE's test split has no labels (all -1), so validation is our eval set.
    train_loader = DataLoader(
        dataset["train"], batch_size=cfg.batch_size, shuffle=True, **common
    )
    val_loader = DataLoader(
        dataset["validation"], batch_size=cfg.batch_size * 2, shuffle=False, **common
    )
    return train_loader, val_loader


# %% Optimizers
def make_frozen_optimizer(model: SentimentClassificationBert, cfg: Config):
    return torch.optim.AdamW(
        model.classifier.parameters(),
        lr=cfg.frozen_head_lr,
        weight_decay=cfg.weight_decay,
    )


def make_finetune_optimizer(model: SentimentClassificationBert, cfg: Config):
    # BERT gets a small LR so it isn't wrecked; the already-warm head moves faster.
    return torch.optim.AdamW(
        [
            {"params": model.bert.parameters(), "lr": cfg.backbone_lr},
            {"params": model.classifier.parameters(), "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )


# %% Trainer
class Trainer:
    def __init__(
        self,
        model: SentimentClassificationBert,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: Config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()
        self.epoch = 0  # global epoch counter across phases
        self.best_val_acc = 0.0
        self.ckpt_path = Path(cfg.output_dir) / "best_model.pt"

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, n = 0.0, 0, 0
        for batch in self.val_loader:
            batch = self._to_device(batch)
            logits = self.model(batch["input_ids"], batch["attention_mask"])
            labels = batch["labels"]
            total_loss += self.loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            n += labels.size(0)
        return total_loss / n, correct / n

    def train_one_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        phase: str = "",
    ) -> float:
        self.model.train()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        running, epoch_total = 0.0, 0.0
        n_steps = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, start=1):
            batch = self._to_device(batch)
            logits = self.model(batch["input_ids"], batch["attention_mask"])
            loss = self.loss_fn(logits, batch["labels"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, self.cfg.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running += loss.item()
            epoch_total += loss.item()
            if step % self.cfg.log_every == 0:
                lrs = ", ".join(f"{g['lr']:.2e}" for g in optimizer.param_groups)
                print(
                    f"[{phase}] epoch {self.epoch} | step {step}/{n_steps} | "
                    f"train loss {running / self.cfg.log_every:.4f} | lr {lrs}"
                )
                running = 0.0
        return epoch_total / n_steps

    def run_phase(
        self,
        phase: str,
        num_epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> None:
        print(f"\n=== Phase: {phase} | {num_epochs} epoch(s) | {count_params(self.model)}")
        for _ in range(num_epochs):
            self.epoch += 1
            train_loss = self.train_one_epoch(optimizer, scheduler, phase)
            val_loss, val_acc = self.evaluate()
            print(
                f"== [{phase}] epoch {self.epoch} | train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | val acc {val_acc:.4f}"
            )
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self.ckpt_path)
                print(f"   saved new best ({val_acc:.4f}) -> {self.ckpt_path}")


# %% Main
def main(cfg: Optional[Config] = None) -> SentimentClassificationBert:
    cfg = cfg or Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_loader, val_loader = build_dataloaders(cfg, tokenizer)

    model = SentimentClassificationBert(
        cfg.model_name, hidden_dims=cfg.hidden_dims, dropout=cfg.dropout
    ).to(device)
    trainer = Trainer(model, train_loader, val_loader, device, cfg)

    # Phase 1: frozen BERT, warm up the randomly initialised head.
    model.freeze_backbone()
    trainer.run_phase("frozen", cfg.frozen_epochs, make_frozen_optimizer(model, cfg))

    # Phase 2: unfreeze BERT and fine-tune everything with warmup + linear decay.
    model.unfreeze_backbone()
    optimizer = make_finetune_optimizer(model, cfg)
    total_steps = cfg.finetune_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    trainer.run_phase("finetune", cfg.finetune_epochs, optimizer, scheduler)

    print(f"\nbest val acc: {trainer.best_val_acc:.4f}")
    return model


if __name__ == "__main__":
    main()