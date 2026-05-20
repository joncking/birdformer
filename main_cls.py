import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]")
TRAIN_SPLIT = 0.8
CHECKPOINT_DIR = Path("models")
PLOTS_DIR = Path("plots")


def plot_training_history(history, save_path=PLOTS_DIR / "training_history.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["test_loss"], label="test")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["accuracy"], label="accuracy")
    axes[1].plot(epochs, history["balanced_acc"], label="balanced acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("Test accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["lr"])
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_title("Learning rate")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")

class Tokenizer():
    def __init__(self, seq_len, vocabulary_path="data/vocabulary.json") -> None:
        self.token_to_id = json.loads(Path(vocabulary_path).read_text())
        self.id_to_token = {token_id: token for token, token_id in self.token_to_id.items()}
        self.pad_id = self.token_to_id["[pad]"]
        self.unk_id = self.token_to_id["[unk]"]
        self.eos_id = self.token_to_id["[eos]"]
        self.cls_id = self.token_to_id["[cls]"]
        self.vocab_size = len(self.token_to_id)
        self.seq_len = seq_len

    def split_tokens(self, text):
        return TOKEN_PATTERN.findall(text)

    def encode(self, text) -> torch.Tensor:
        tokens = ["[cls]"] + self.split_tokens(text) + ["[eos]"]
        token_ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        token_ids = token_ids[:self.seq_len]
        token_ids += [self.pad_id] * (self.seq_len - len(token_ids))
        return torch.LongTensor(token_ids)

    def decode(self, token_ids):
        return [self.id_to_token[int(token_id)] for token_id in token_ids]

    def load_chunks(self, chunks_path="data/chunks.txt") -> torch.Tensor:
        rows = Path(chunks_path).read_text().splitlines()
        chunks = [[int(token_id) for token_id in row.split()] for row in rows]
        return torch.LongTensor(chunks)

    def load_chunk_labels(self, labels_path="data/chunk_labels.txt") -> torch.Tensor:
        label_to_id = {"reptile": 0, "bird": 1}
        labels = Path(labels_path).read_text().splitlines()
        return torch.LongTensor([label_to_id[label] for label in labels])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embedding_layer(x)

class PositionEmbedding(nn.Module):
    def __init__(self, seq_len, d_model) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embedding_layer = nn.Embedding(seq_len, d_model)
    def forward(self, x):
        assert x.shape[1] == self.seq_len
        positions = torch.arange(self.seq_len, device=x.device)
        return self.embedding_layer(positions)

class Attention(nn.Module):
    def __init__(self, seq_len, d_model, d_head, num_heads, dropout=0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(d_model, d_head * 3 * num_heads)
        self.output_layer = nn.Linear(self.num_heads * d_head, d_model)
        self.seq_len = seq_len
        self.d_head = d_head
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    def forward(self, x):
        batch = x.shape[0]
        assert x.shape[1] == self.seq_len
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, self.seq_len, 3, self.num_heads, self.d_head)
        qkv = torch.transpose(qkv, 1, 3)
        q, k, v = torch.unbind(qkv, dim=2)
        attention_weights = F.softmax(q @ torch.transpose(k, 2, 3)/ math.sqrt(self.d_head), dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        scores = attention_weights @ v
        weights = torch.transpose(scores, 1, 2)
        weights = weights.reshape(batch, self.seq_len, self.num_heads * self.d_head)
        outputs = self.output_layer(weights)
        return self.proj_dropout(outputs)

class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    def forward(self, x):
        average = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        return self.gamma * (x - average)/torch.sqrt(var + 1e-5) + self.beta


class EncoderBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_head, num_heads=3):
        super().__init__()
        self.attention = Attention(seq_len, d_model, d_head, num_heads)
        self.layer_norm1 = LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.layer_norm2 = LayerNorm(d_model)
    def forward(self, x):
        attention_out = self.attention(x)
        norm1_out = self.layer_norm1(x + attention_out)
        ffn_out = self.ffn(norm1_out)
        norm2_out = self.layer_norm2(ffn_out + norm1_out)
        return norm2_out

class ClassificationEncoder(nn.Module):
    def __init__(
            self, 
            vocab_size,
            seq_len, 
            d_model,
            d_head,
            num_heads=3, 
            num_classes = 2, 
            num_encoders = 2
            ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionEmbedding(seq_len, d_model)
        self.encoders = nn.ModuleList([
            EncoderBlock(seq_len, d_model, d_head, num_heads)
            for i in range(num_encoders)
        ])
        self.task_head = nn.Linear(d_model, num_classes)
    def forward(self, x):
        assert x.shape[1] == self.seq_len
        tok_e = self.token_embedding(x)
        pos_e = self.position_embedding(x)
        x = tok_e + pos_e
        for encoder in self.encoders:
            x = encoder(x)
        cls_vector = x[:, 0, :]
        logits = self.task_head(cls_vector)
        return logits

def save_checkpoint(
        path,
        model,
        optimizer,
        model_config,
        train_config,
        loss=None,
        step=None,
        ):
    path.parent.mkdir(exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "train_config": train_config,
        "loss": loss,
        "step": step,
    }
    torch.save(checkpoint, path)

if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    tokenizer = Tokenizer(seq_len=128)
    chunks = tokenizer.load_chunks('data/chunks.txt')
    labels = tokenizer.load_chunk_labels('data/chunk_labels.txt')

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "seq_len": tokenizer.seq_len,
        "d_model": 512,
        "d_head": 128,
        "num_heads": 6,
        "num_classes": 2,
        "num_encoders": 2,
    }
    model = ClassificationEncoder(**model_config)
    learning_rate = 0.00005
    train_config = {
        "train_split": TRAIN_SPLIT,
        "seed": 42,
        "batch_size": 64,
        "learning_rate": learning_rate,
    }
    num_classes = 2
    model.train()

    model.to(device)
    chunks = chunks.to(device)
    labels = labels.to(device)

    num_chunks = chunks.shape[0]

    rng = np.random.default_rng(seed=42)

    indices = list(range(num_chunks))
    indices = rng.permutation(indices)

    train_partition = int(num_chunks * TRAIN_SPLIT)
    train_x = chunks[indices[:train_partition]]
    train_y = labels[indices[:train_partition]]
    test_x = chunks[indices[train_partition:]]
    test_y = labels[indices[train_partition:]]

    class_counts = torch.bincount(train_y, minlength=num_classes).float()
    class_weights = len(train_y) / (num_classes * class_counts)
    print(f"train class counts: {class_counts.tolist()}  weights: {class_weights.tolist()}")
    #loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    batch_size = 64
    half_batch = batch_size // 2
    last_loss = None
    num_epochs = 40

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    class_pools = [(train_y == c).nonzero(as_tuple=True)[0] for c in range(num_classes)]
    batches_per_epoch = max(len(p) for p in class_pools) // half_batch
    sampler_gen = torch.Generator(device=device).manual_seed(42)

    def epoch_class_sequence(pool, total):
        out = []
        remaining = total
        while remaining > 0:
            perm = pool[torch.randperm(len(pool), generator=sampler_gen, device=device)]
            take = min(len(perm), remaining)
            out.append(perm[:take])
            remaining -= take
        return torch.cat(out)

    history = {"train_loss": [], "test_loss": [], "accuracy": [], "balanced_acc": [], "lr": []}

    for epoch in range(num_epochs):
        print(f"--- STARTING EPOCH {epoch+1}/{num_epochs} ---")
        per_class_total = batches_per_epoch * half_batch
        class_sequences = [epoch_class_sequence(p, per_class_total) for p in class_pools]

        model.train()
        train_loss_sum, train_count = 0.0, 0
        for b in range(batches_per_epoch):
            s = b * half_batch
            batch_idx = torch.cat([seq[s:s + half_batch] for seq in class_sequences])
            batch_idx = batch_idx[torch.randperm(len(batch_idx), generator=sampler_gen, device=device)]
            batch_x = train_x[batch_idx]
            batch_y = train_y[batch_idx]

            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            last_loss = loss.item()
            train_loss_sum += last_loss * len(batch_x)
            train_count += len(batch_x)

            if b % 100 == 0:
                seen = min((b + 1) * batch_size, batches_per_epoch * batch_size)
                print(f"loss: {last_loss:>7f}  [{seen:>5d}/{batches_per_epoch * batch_size:>5d}]   LR: {scheduler.get_last_lr()[0]}")
        avg_train_loss = train_loss_sum / train_count
        
        model.eval()

        test_loss, correct = 0, 0
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
        with torch.no_grad():
            for i in range (0, len(test_x), batch_size):
                batch_x = test_x[i: i + batch_size]
                batch_y = test_y[i: i + batch_size]
                pred = model(batch_x)
                test_loss += loss_fn(pred, batch_y).item() * len(batch_x)
                pred_labels = pred.argmax(1)
                correct += (pred_labels == batch_y).type(torch.float).sum().item()
                idx = batch_y * num_classes + pred_labels
                confusion += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        test_loss /= len(test_x)
        correct /= len(test_x)
        per_class_total = confusion.sum(dim=1).clamp(min=1)
        per_class_recall = confusion.diag().float() / per_class_total.float()
        balanced_acc = per_class_recall.mean().item()
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Balanced acc: {(100*balanced_acc):>0.1f}%, Avg train loss: {avg_train_loss:>8f}, Avg test loss: {test_loss:>8f} \n")

        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(test_loss)
        history["accuracy"].append(correct)
        history["balanced_acc"].append(balanced_acc)
        history["lr"].append(scheduler.get_last_lr()[0])

        class_names = ["reptile", "bird"]
        cm = confusion.cpu().tolist()
        col_w = max(len(n) for n in class_names + [str(v) for row in cm for v in row]) + 1
        header = " " * (col_w + 8) + "pred"
        col_labels = " " * (col_w + 2) + "".join(f"{n:>{col_w}}" for n in class_names)
        print(header)
        print(col_labels)
        for i, name in enumerate(class_names):
            prefix = "true " if i == 0 else "     "
            row = "".join(f"{v:>{col_w}}" for v in cm[i])
            print(f"{prefix}{name:>{col_w}}{row}")
        print()
        scheduler.step()

    save_checkpoint(
        CHECKPOINT_DIR / "classification_encoder.pt",
        model,
        optimizer,
        model_config,
        train_config,
        loss=last_loss,
        step=train_partition,
    )

    plot_training_history(history)
