"""Interactive marimo notebook for exploring CLS embeddings.

Run with:
    uv run marimo edit embedding_explorer.py
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    from pathlib import Path

    import altair as alt
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    from embedding_viz import (
        CLASS_NAMES,
        CHECKPOINT_PATH,
        extract_cls_embeddings,
        load_model,
    )
    from main_cls import Tokenizer
    return (
        CHECKPOINT_PATH,
        CLASS_NAMES,
        PCA,
        Path,
        TSNE,
        Tokenizer,
        alt,
        extract_cls_embeddings,
        load_model,
        np,
        pd,
        torch,
    )


@app.cell
def __(CHECKPOINT_PATH, Tokenizer, extract_cls_embeddings, load_model, torch):
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    model, cfg = load_model(CHECKPOINT_PATH, device)
    tokenizer = Tokenizer(seq_len=cfg["seq_len"])
    chunks = tokenizer.load_chunks("data/chunks.txt").to(device)
    labels = tokenizer.load_chunk_labels("data/chunk_labels.txt").numpy()
    embeddings = extract_cls_embeddings(model, chunks)
    return chunks, embeddings, labels, model, tokenizer


@app.cell
def __(CLASS_NAMES, chunks, labels, tokenizer):
    def chunk_to_text(chunk_ids, max_tokens=40):
        toks = tokenizer.decode(chunk_ids.cpu().tolist())
        toks = [t for t in toks if t not in ("[pad]", "[cls]", "[eos]")]
        text = " ".join(toks[:max_tokens])
        if len(toks) > max_tokens:
            text += " ..."
        return text

    texts = [chunk_to_text(c) for c in chunks]
    class_names = [CLASS_NAMES[i] for i in labels]
    return chunk_to_text, class_names, texts


@app.cell
def __(mo):
    method = mo.ui.dropdown(
        options=["PCA", "t-SNE"],
        value="PCA",
        label="Projection",
    )
    method
    return (method,)


@app.cell
def __(PCA, TSNE, embeddings, method, np):
    if method.value == "PCA":
        coords = PCA(n_components=2).fit_transform(embeddings)
    else:
        pca_50 = PCA(n_components=min(50, embeddings.shape[1])).fit_transform(embeddings)
        coords = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            random_state=42,
        ).fit_transform(pca_50)
    coords = np.asarray(coords)
    return (coords,)


@app.cell
def __(class_names, coords, pd, texts):
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "class": class_names,
        "text": texts,
        "idx": range(len(texts)),
    })
    return (df,)


@app.cell
def __(alt, df, method, mo):
    chart = (
        alt.Chart(df)
        .mark_circle(size=40, opacity=0.6)
        .encode(
            x=alt.X("x:Q", title=f"{method.value} dim 1"),
            y=alt.Y("y:Q", title=f"{method.value} dim 2"),
            color=alt.Color("class:N", scale=alt.Scale(scheme="category10")),
            tooltip=[
                alt.Tooltip("idx:Q", title="chunk #"),
                alt.Tooltip("class:N"),
                alt.Tooltip("text:N", title="text"),
            ],
        )
        .properties(width=720, height=520, title=f"CLS embeddings — {method.value}")
        .interactive()
    )
    chart_ui = mo.ui.altair_chart(chart)
    chart_ui
    return chart, chart_ui


@app.cell
def __(chart_ui, mo, pd):
    selected = chart_ui.value if chart_ui.value is not None else pd.DataFrame()
    mo.md(
        f"**{len(selected)} points selected**\n\n"
        + (
            "\n\n".join(
                f"- *[{row['class']}]* {row['text']}"
                for _, row in selected.head(20).iterrows()
            )
            if len(selected)
            else "_Box-select points on the chart to see their full text here._"
        )
    )
    return (selected,)


if __name__ == "__main__":
    app.run()
