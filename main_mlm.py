import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]")
TRAIN_SPLIT = 0.8
CHECKPOINT_DIR = Path("models")
PLOTS_DIR = Path("plots")


def plot_training_history(history, save_path=PLOTS_DIR / "mlm_training_history.png"):
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

    axes[1].plot(epochs, history["masked_accuracy"], label="masked accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Masked Token Accuracy")
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
        self.mask_id = self.token_to_id["[mask]"]
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

class MLMMasker():
    def __init__(
            self,
            mask_id,
            pad_id,
            cls_id,
            eos_id,
            mask_prob=0.15,
            ignore_index=-100,
            ) -> None:
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.eos_id = eos_id
        self.mask_prob = mask_prob
        self.ignore_index = ignore_index

    def __call__(self, chunks, generator=None):
        eligible = (
            (chunks != self.pad_id)
            & (chunks != self.cls_id)
            & (chunks != self.eos_id)
        )
        chunk_mask = (
            torch.rand(chunks.shape, device=chunks.device, generator=generator) < self.mask_prob
        ) & eligible

        if not chunk_mask.any():
            eligible_positions = eligible.nonzero(as_tuple=False)
            if len(eligible_positions) > 0:
                choice = torch.randint(len(eligible_positions), (1,), device=chunks.device, generator=generator)
                row, col = eligible_positions[choice.item()]
                chunk_mask[row, col] = True

        masked_chunks = chunks.clone()
        masked_chunks[chunk_mask] = self.mask_id

        targets = chunks.clone()
        targets[~chunk_mask] = self.ignore_index
        return masked_chunks, targets, chunk_mask


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
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.Dropout(p=0.1),
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

class MLMEncoder(nn.Module):
    def __init__(
            self, 
            vocab_size,
            seq_len, 
            d_model,
            d_head,
            num_heads=3, 
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
        self.task_head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        assert x.shape[1] == self.seq_len
        tok_e = self.token_embedding(x)
        pos_e = self.position_embedding(x)
        x = tok_e + pos_e
        for encoder in self.encoders:
            x = encoder(x)
        logits = self.task_head(x)
        return logits

def masked_language_model_loss(logits, targets, ignore_index=-100):
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )

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
    masker = MLMMasker(
        mask_id=tokenizer.mask_id,
        pad_id=tokenizer.pad_id,
        cls_id=tokenizer.cls_id,
        eos_id=tokenizer.eos_id,
        mask_prob=0.15,
    )

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "seq_len": tokenizer.seq_len,
        "d_model": 512,
        "d_head": 128,
        "num_heads": 6,
        "num_encoders": 2,
    }
    model = MLMEncoder(**model_config)
    batch_size = 64
    learning_rate = 0.0005
    train_config = {
        "train_split": TRAIN_SPLIT,
        "seed": 42,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "mask_prob": masker.mask_prob,
        "ignore_index": masker.ignore_index,
    }
    model.train()

    model.to(device)
    chunks = chunks.to(device)

    num_chunks = chunks.shape[0]
    split_gen = torch.Generator(device=device).manual_seed(42)
    indices = torch.randperm(num_chunks, generator=split_gen, device=device)

    train_partition = int(num_chunks * TRAIN_SPLIT)
    train_x = chunks[indices[:train_partition]]
    test_x = chunks[indices[train_partition:]]

    last_loss = None
    num_epochs = 40

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    sampler_gen = torch.Generator(device=device).manual_seed(42)
    global_step = 0

    history = {"train_loss": [], "test_loss": [], "masked_accuracy": [], "lr": []}

    for epoch in range(num_epochs):
        print(f"--- STARTING EPOCH {epoch+1}/{num_epochs} ---")
        epoch_order = torch.randperm(len(train_x), generator=sampler_gen, device=device)

        model.train()
        train_loss_sum, train_count = 0.0, 0
        for b, start in enumerate(range(0, len(train_x), batch_size)):
            batch_idx = epoch_order[start:start + batch_size]
            batch_x = train_x[batch_idx]
            masked_x, targets, chunk_mask = masker(batch_x, generator=sampler_gen)

            pred = model(masked_x)
            loss = masked_language_model_loss(pred, targets, ignore_index=masker.ignore_index)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            last_loss = loss.item()
            masked_count = chunk_mask.sum().item()
            train_loss_sum += last_loss * masked_count
            train_count += masked_count

            if b % 100 == 0:
                seen = min(start + len(batch_x), len(train_x))
                print(f"loss: {last_loss:>7f}  [{seen:>5d}/{len(train_x):>5d}]   LR: {scheduler.get_last_lr()[0]}")
        avg_train_loss = train_loss_sum / train_count
        
        model.eval()

        test_loss_sum, test_count, correct = 0.0, 0, 0
        eval_gen = torch.Generator(device=device).manual_seed(1000)
        with torch.no_grad():
            for i in range (0, len(test_x), batch_size):
                batch_x = test_x[i: i + batch_size]
                masked_x, targets, chunk_mask = masker(batch_x, generator=eval_gen)
                pred = model(masked_x)
                loss = masked_language_model_loss(pred, targets, ignore_index=masker.ignore_index)
                masked_count = chunk_mask.sum().item()
                test_loss_sum += loss.item() * masked_count
                test_count += masked_count
                pred_labels = pred.argmax(dim=-1)
                correct += (pred_labels[chunk_mask] == targets[chunk_mask]).type(torch.float).sum().item()
        test_loss = test_loss_sum / test_count
        masked_accuracy = correct / test_count
        print(f"Test Error: \n Masked accuracy: {(100*masked_accuracy):>0.1f}%, Avg train loss: {avg_train_loss:>8f}, Avg test loss: {test_loss:>8f} \n")

        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(test_loss)
        history["masked_accuracy"].append(masked_accuracy)
        history["lr"].append(scheduler.get_last_lr()[0])

        scheduler.step()

    save_checkpoint(
        CHECKPOINT_DIR / "mlm_encoder.pt",
        model,
        optimizer,
        model_config,
        train_config,
        loss=last_loss,
        step=global_step,
    )

    plot_training_history(history)
