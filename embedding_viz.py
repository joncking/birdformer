"""Extract CLS embeddings from a trained classifier and visualize them in 2D.

Usage:
    uv run embedding_viz.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from main_cls import ClassificationEncoder, Tokenizer

CHECKPOINT_PATH = Path("models/classification_encoder.pt")
PLOTS_DIR = Path("plots")
CLASS_NAMES = ["reptile", "bird"]

# The checkpoint's saved model_config can drift from the actual saved weights
# (e.g. you changed d_model in the model build but forgot the config dict).
# We derive everything we can from state_dict shapes; the only thing we can't
# infer is the num_heads / d_head split (only their product is in the weights),
# so set those here if the checkpoint config is wrong.
NUM_HEADS_OVERRIDE = 6
D_HEAD_OVERRIDE = 128


def derive_config(state_dict, fallback_config):
    """Build a model config from state_dict shapes, falling back to saved config."""
    cfg = dict(fallback_config)
    tok_w = state_dict["token_embedding.embedding_layer.weight"]
    pos_w = state_dict["position_embedding.embedding_layer.weight"]
    task_w = state_dict["task_head.weight"]
    out_w = state_dict["encoders.0.attention.output_layer.weight"]

    cfg["vocab_size"] = tok_w.shape[0]
    cfg["d_model"] = tok_w.shape[1]
    cfg["seq_len"] = pos_w.shape[0]
    cfg["num_classes"] = task_w.shape[0]
    cfg["num_encoders"] = 1 + max(
        int(k.split(".")[1]) for k in state_dict if k.startswith("encoders.")
    )

    if NUM_HEADS_OVERRIDE is not None:
        cfg["num_heads"] = NUM_HEADS_OVERRIDE
    if D_HEAD_OVERRIDE is not None:
        cfg["d_head"] = D_HEAD_OVERRIDE

    expected = cfg["num_heads"] * cfg["d_head"]
    actual = out_w.shape[1]
    if expected != actual:
        raise RuntimeError(
            f"num_heads*d_head={expected} but weights need {actual}. "
            f"Set NUM_HEADS_OVERRIDE / D_HEAD_OVERRIDE at the top of this file."
        )
    return cfg


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = derive_config(ckpt["model_state_dict"], ckpt["model_config"])
    print(f"loaded config: {cfg}")
    model = ClassificationEncoder(
        vocab_size=cfg["vocab_size"],
        seq_len=cfg["seq_len"],
        d_model=cfg["d_model"],
        d_head=cfg["d_head"],
        num_heads=cfg["num_heads"],
        num_classes=cfg["num_classes"],
        num_encoders=cfg["num_encoders"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def extract_cls_embeddings(model, chunks, batch_size=64):
    """Run chunks through the encoder, return the CLS vector for each."""
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        x = model.token_embedding(batch) + model.position_embedding(batch)
        for encoder in model.encoders:
            x = encoder(x)
        embeddings.append(x[:, 0, :].cpu())
    return torch.cat(embeddings).numpy()


def plot_2d(coords_2d, labels, title, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for class_id, name in enumerate(CLASS_NAMES):
        mask = labels == class_id
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            s=12,
            alpha=0.6,
            label=name,
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")


def nearest_neighbors(embeddings, query_idx, k=5):
    """Return indices of the k nearest neighbors of query_idx by cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-8, None)
    sims = normed @ normed[query_idx]
    order = np.argsort(-sims)
    return [i for i in order if i != query_idx][:k]


if __name__ == "__main__":
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    model, cfg = load_model(CHECKPOINT_PATH, device)
    tokenizer = Tokenizer(seq_len=cfg["seq_len"])

    chunks = tokenizer.load_chunks("data/chunks.txt").to(device)
    labels = tokenizer.load_chunk_labels("data/chunk_labels.txt").numpy()

    print(f"Extracting CLS embeddings for {len(chunks)} chunks...")
    embeddings = extract_cls_embeddings(model, chunks)
    print(f"Embedding shape: {embeddings.shape}")

    pca_2d = PCA(n_components=2).fit_transform(embeddings)
    plot_2d(pca_2d, labels, "CLS embeddings — PCA", PLOTS_DIR / "embeddings_pca.png")

    print("Running t-SNE (this takes a moment)...")
    pca_50 = PCA(n_components=min(50, embeddings.shape[1])).fit_transform(embeddings)
    tsne_2d = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(pca_50)
    plot_2d(tsne_2d, labels, "CLS embeddings — t-SNE", PLOTS_DIR / "embeddings_tsne.png")

    print("\nNearest-neighbor sanity check (cosine):")
    raw_lines = Path("data/chunks.txt").read_text().splitlines()
    for query_idx in [0, 100, 500]:
        if query_idx >= len(embeddings):
            continue
        nbrs = nearest_neighbors(embeddings, query_idx, k=5)
        query_tokens = " ".join(tokenizer.decode(chunks[query_idx].cpu().tolist())[:15])
        print(f"\n  query [{CLASS_NAMES[labels[query_idx]]}]: {query_tokens}...")
        for n in nbrs:
            n_tokens = " ".join(tokenizer.decode(chunks[n].cpu().tolist())[:15])
            print(f"    -> [{CLASS_NAMES[labels[n]]}] {n_tokens}...")
