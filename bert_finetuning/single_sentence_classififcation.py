# %%
from datasets import load_dataset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
# %%
MODEL_NAME = "google-bert/bert-base-uncased"

task = "sst2" # sentiment classification
dataset = load_dataset("nyu-mll/glue", task)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %% Model
class SentimentClassificationBert(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.bert.requires_grad_(False)
        self.bert.eval()
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen encoder's dropout disabled while the head trains.
        self.bert.eval()
        return self

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        # Return raw logits; CrossEntropyLoss applies log-softmax internally.
        return self.classifier(cls_embedding)

# %% Data: collate + dataloaders
BATCH_SIZE = 32
MAX_LENGTH = 128


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Tokenize raw SST-2 examples and pad to the longest sentence in the batch."""
    enc = tokenizer(
        [ex["sentence"] for ex in batch],
        padding=True,  # dynamic padding: only as long as this batch needs
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor([ex["label"] for ex in batch], dtype=torch.long),
    }


# GLUE's test split has no labels (all -1), so validation is our eval set.
train_loader = DataLoader(
    dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    dataset["validation"], batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=collate_fn
)

# %% Training setup
NUM_EPOCHS = 3
LR = 1e-3  # a linear probe on frozen features tolerates a much higher LR than full fine-tuning
LOG_EVERY = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentClassificationBert().to(device)
# Only the head is trainable, so only hand the head to the optimizer.
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"device={device} | trainable params: {n_trainable:,} / {n_total:,}")

# %% Train / eval loop
def to_device(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        batch = to_device(batch)
        logits = model(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"]
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


def train_one_epoch(model: nn.Module, loader: DataLoader, epoch: int) -> float:
    model.train()
    running, epoch_total = 0.0, 0.0
    for step, batch in enumerate(loader, start=1):
        batch = to_device(batch)
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = loss_fn(logits, batch["labels"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += loss.item()
        epoch_total += loss.item()
        if step % LOG_EVERY == 0:
            print(f"epoch {epoch} | step {step}/{len(loader)} | train loss {running / LOG_EVERY:.4f}")
            running = 0.0
    return epoch_total / len(loader)


for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, epoch)
    val_loss, val_acc = evaluate(model, val_loader)
    print(
        f"== epoch {epoch} | train loss {train_loss:.4f} | "
        f"val loss {val_loss:.4f} | val acc {val_acc:.4f}"
    )

# %%
