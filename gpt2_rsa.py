"""RSA on GPT-2 hidden states over bird vs. reptile paragraphs.

Standalone from the from-scratch transformer in this repo; it only reuses the
cleaned paragraph files under data/.

Run with:
    uv run marimo edit gpt2_rsa.py
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import numpy as np
    import pandas as pd
    import torch
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import rankdata, spearmanr
    from sklearn.feature_extraction.text import TfidfVectorizer
    from transformers import GPT2Model, GPT2TokenizerFast

    alt.data_transformers.disable_max_rows()  # RDM heatmaps are n² cells
    return (
        GPT2Model,
        GPT2TokenizerFast,
        Path,
        TfidfVectorizer,
        alt,
        np,
        pd,
        pdist,
        rankdata,
        spearmanr,
        squareform,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # GPT-2 × RSA: birds vs. reptiles

    Representational similarity analysis: embed each paragraph with GPT-2,
    build a **representational dissimilarity matrix** (RDM) per layer, then
    compare those RDMs (Spearman on the upper triangles) to

    - a **category model** RDM — 0 if two paragraphs share a class, 1 otherwise
    - a **lexical model** RDM — cosine distance between TF-IDF vectors, as a
      "is this just word overlap?" baseline
    - each other (layer × layer)
    """)
    return


@app.cell
def _(mo):
    n_per_class = mo.ui.slider(20, 300, value=100, step=10, label="Paragraphs per class")
    min_words = mo.ui.slider(10, 150, value=40, step=5, label="Min words per paragraph")
    max_tokens = mo.ui.slider(32, 512, value=128, step=32, label="Max GPT-2 tokens")
    pooling = mo.ui.dropdown(["mean", "last"], value="mean", label="Token pooling")
    metric = mo.ui.dropdown(
        ["correlation", "cosine", "euclidean"], value="correlation", label="RDM distance"
    )
    seed = mo.ui.number(0, 10_000, value=0, label="Sample seed")
    mo.vstack([
        mo.hstack([n_per_class, min_words, seed]),
        mo.hstack([max_tokens, pooling, metric]),
    ])
    return max_tokens, metric, min_words, n_per_class, pooling, seed


@app.cell
def _(Path, min_words, n_per_class, np, pd, seed):
    def load_paragraphs(path):
        text = Path(path).read_text(encoding="utf-8")
        paras = [" ".join(p.split()) for p in text.split("\n\n")]
        return [p for p in paras if len(p.split()) >= min_words.value]

    _rng = np.random.default_rng(seed.value)
    _rows = []
    for _label, _path in [("bird", "data/birds_paragraphs.txt"), ("reptile", "data/reptiles_paragraphs.txt")]:
        _paras = load_paragraphs(_path)
        _k = min(n_per_class.value, len(_paras))
        for _i in _rng.choice(len(_paras), size=_k, replace=False):
            _rows.append({"label": _label, "text": _paras[_i]})

    # Sorted by class so the RDM heatmaps show block structure.
    items = pd.DataFrame(_rows).reset_index(names="idx")
    items["label"].value_counts()
    return (items,)


@app.cell
def _(GPT2Model, GPT2TokenizerFast, torch):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    gpt2 = GPT2Model.from_pretrained("gpt2").to(device).eval()
    return device, gpt2, tok


@app.cell
def _(device, gpt2, items, max_tokens, mo, np, pooling, tok, torch):
    @torch.no_grad()
    def embed_all_layers(texts, batch_size=16):
        """Returns (n_layers + 1, n_texts, d_model); index 0 is the embedding layer."""
        out = []
        for _start in mo.status.progress_bar(
            range(0, len(texts), batch_size), title="Running GPT-2"
        ):
            batch = tok(
                texts[_start : _start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens.value,
            ).to(device)
            hs = torch.stack(gpt2(**batch, output_hidden_states=True).hidden_states)  # (L, B, T, D)
            mask = batch["attention_mask"]
            if pooling.value == "mean":
                m = mask[None, :, :, None].to(hs.dtype)
                pooled = (hs * m).sum(2) / m.sum(2)
            else:  # last real token (right padding)
                last = mask.sum(1) - 1
                pooled = hs[:, torch.arange(hs.shape[1]), last]
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=1)

    reps = embed_all_layers(items["text"].tolist())
    n_layers = reps.shape[0]
    mo.md(f"Hidden states: `{reps.shape}` (layers × paragraphs × dims)")
    return n_layers, reps


@app.cell
def _(TfidfVectorizer, items, metric, np, pdist, reps):
    # Condensed (upper-triangle) RDMs.
    layer_rdms = np.stack([pdist(r, metric=metric.value) for r in reps])

    _y = (items["label"] == "bird").to_numpy()[:, None].astype(float)
    category_rdm = pdist(_y, metric="cityblock")  # 0 same class, 1 different

    _tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(items["text"])
    lexical_rdm = pdist(_tfidf.toarray(), metric="cosine")
    return category_rdm, layer_rdms, lexical_rdm


@app.cell
def _(category_rdm, layer_rdms, lexical_rdm, n_layers, pd, spearmanr):
    def rsa(a, b):
        return spearmanr(a, b).statistic

    rsa_df = pd.DataFrame(
        [
            {"layer": _l, "model": _name, "spearman": rsa(layer_rdms[_l], _ref)}
            for _l in range(n_layers)
            for _name, _ref in [("category", category_rdm), ("lexical (TF-IDF)", lexical_rdm)]
        ]
    )
    return rsa, rsa_df


@app.cell
def _(alt, mo, rsa_df):
    _chart = (
        alt.Chart(rsa_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("layer:O", title="GPT-2 layer (0 = embeddings)"),
            y=alt.Y("spearman:Q", title="Spearman ρ with model RDM"),
            color=alt.Color("model:N", title="Model RDM"),
            tooltip=["layer", "model", alt.Tooltip("spearman:Q", format=".3f")],
        )
        .properties(width=640, height=320, title="RSA across layers")
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(layer_rdms, n_layers, np, rsa):
    layer_sim = np.ones((n_layers, n_layers))
    for _i in range(n_layers):
        for _j in range(_i + 1, n_layers):
            layer_sim[_i, _j] = layer_sim[_j, _i] = rsa(layer_rdms[_i], layer_rdms[_j])
    return (layer_sim,)


@app.cell
def _(alt, layer_sim, n_layers, pd):
    _df = pd.DataFrame(
        [
            {"layer_a": _i, "layer_b": _j, "spearman": layer_sim[_i, _j]}
            for _i in range(n_layers)
            for _j in range(n_layers)
        ]
    )
    (
        alt.Chart(_df)
        .mark_rect()
        .encode(
            x=alt.X("layer_a:O", title="layer"),
            y=alt.Y("layer_b:O", title="layer", sort="descending"),
            color=alt.Color("spearman:Q", scale=alt.Scale(scheme="viridis"), title="ρ"),
            tooltip=["layer_a", "layer_b", alt.Tooltip("spearman:Q", format=".3f")],
        )
        .properties(width=360, height=360, title="Layer × layer RDM similarity")
    )
    return


@app.cell
def _(mo, n_layers):
    layer_pick = mo.ui.slider(0, n_layers - 1, value=n_layers // 2, label="Inspect layer")
    n_perm = mo.ui.slider(0, 2000, value=500, step=100, label="Permutations")
    mo.hstack([layer_pick, n_perm])
    return layer_pick, n_perm


@app.cell
def _(alt, items, layer_pick, layer_rdms, lexical_rdm, mo, pd, squareform):
    def rdm_heatmap(condensed, title):
        _sq = squareform(condensed)
        _n = _sq.shape[0]
        _df = pd.DataFrame(
            {
                "i": [i for i in range(_n) for _ in range(_n)],
                "j": [j for _ in range(_n) for j in range(_n)],
                "d": _sq.ravel(),
            }
        )
        return (
            alt.Chart(_df)
            .mark_rect()
            .encode(
                x=alt.X("i:O", axis=None),
                y=alt.Y("j:O", axis=None),
                color=alt.Color("d:Q", scale=alt.Scale(scheme="magma", reverse=True), title="dissim."),
            )
            .properties(width=300, height=300, title=title)
        )

    _n_bird = int((items["label"] == "bird").sum())
    mo.vstack([
        mo.md(f"Rows/cols sorted by class: first {_n_bird} are birds, the rest reptiles."),
        alt.hconcat(
            rdm_heatmap(layer_rdms[layer_pick.value], f"GPT-2 layer {layer_pick.value} RDM"),
            rdm_heatmap(lexical_rdm, "TF-IDF RDM"),
        ).resolve_scale(color="independent"),
    ])
    return


@app.cell
def _(category_rdm, items, layer_pick, layer_rdms, mo, n_perm, np, pdist, rsa):
    # Permutation test: shuffle class labels, rebuild the category RDM.
    _obs = rsa(layer_rdms[layer_pick.value], category_rdm)
    _rng = np.random.default_rng(0)
    _y = (items["label"] == "bird").to_numpy().astype(float)
    _null = np.array(
        [
            rsa(layer_rdms[layer_pick.value], pdist(_rng.permutation(_y)[:, None], metric="cityblock"))
            for _ in range(n_perm.value)
        ]
    )
    _p = (1 + (_null >= _obs).sum()) / (1 + len(_null)) if len(_null) else float("nan")
    mo.md(
        f"**Layer {layer_pick.value} vs. category:** ρ = {_obs:.3f}, "
        f"permutation p = {_p:.4f} ({n_perm.value} shuffles; null mean {_null.mean() if len(_null) else float('nan'):.3f})"
    )
    return


@app.cell
def _(category_rdm, layer_pick, layer_rdms, lexical_rdm, mo, np, rankdata):
    # Does the layer track category beyond what TF-IDF word overlap explains?
    # Partial Spearman: rank-transform, regress out lexical, correlate residuals.
    def _rank(x):
        return rankdata(x)

    def _resid(y, x):
        X = np.column_stack([np.ones_like(x), x])
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    _a, _c, _lex = _rank(layer_rdms[layer_pick.value]), _rank(category_rdm), _rank(lexical_rdm)
    _partial = np.corrcoef(_resid(_a, _lex), _resid(_c, _lex))[0, 1]
    mo.md(f"**Partial ρ (layer {layer_pick.value} ~ category | lexical):** {_partial:.3f}")
    return


@app.cell
def _(items, mo):
    query = mo.ui.dropdown(
        {f"[{r.label}] {r.text[:80]}…": r.idx for r in items.itertuples()},
        label="Query paragraph",
    )
    query
    return (query,)


@app.cell
def _(items, layer_pick, layer_rdms, mo, np, query, squareform):
    mo.stop(query.value is None, mo.md("_Pick a paragraph to see its nearest neighbours at the inspected layer._"))
    _d = squareform(layer_rdms[layer_pick.value])[query.value]
    _nn = np.argsort(_d)[1:8]
    mo.md(
        f"**Query** *[{items.label[query.value]}]* {items.text[query.value]}\n\n---\n\n"
        + "\n\n".join(
            f"- `{_d[i]:.3f}` *[{items.label[i]}]* {items.text[i][:300]}…" for i in _nn
        )
    )
    return


if __name__ == "__main__":
    app.run()
