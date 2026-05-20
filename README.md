# hwformer

A small transformer learning project built around Louis Figuier's public-domain
Project Gutenberg text, *Reptiles and Birds*.

The point of this repo is not to build a competitive language model. It is an
exercise in building the pieces of a transformer by hand in PyTorch, then using
that tiny stack to classify bird-vs-reptile passages and visualize the resulting
embedding space.

## What Is In Here

- `cleaning.py` cleans the source book text into reptile and bird paragraphs.
- `tokenizer.py` builds a simple word-level vocabulary and fixed-length chunks.
- `main_cls.py` trains an encoder-only classifier with a `[cls]` token head.
- `main_mlm.py` is an MLM version of the same encoder, including a separate
  `MLMMasker` that corrupts batches and scores only masked positions.
- `embedding_viz.py` extracts `[cls]` embeddings from a trained classifier and
  saves PCA/t-SNE plots.
- `embedding_explorer.py` opens an interactive marimo notebook for exploring
  embedding clusters and nearby passages.

The model code intentionally spells out embeddings, multi-head self-attention,
feed-forward blocks, residual connections, layer norm, task heads, and training
loops instead of hiding them behind a high-level transformer library.

## Dataset

The raw source file is Project Gutenberg eBook #45873, *Reptiles and Birds*.
The local pipeline splits the book into reptile and bird sections, removes some
front matter/taxonomy/synonym material, tokenizes the paragraphs, and creates
fixed-length chunks with labels.

Generated data lives under `data/`.

Source text:

> Louis Figuier, *Reptiles and Birds*, edited by Parker Gillmore.
> Project Gutenberg eBook #45873.
> https://www.gutenberg.org/ebooks/45873

Project Gutenberg lists the text as public domain in the USA.

## Running

Install dependencies with `uv`:

```bash
uv sync
```

Prepare the cleaned paragraphs and token chunks:

```bash
uv run python cleaning.py
uv run python tokenizer.py
```

Train the classifier:

```bash
uv run python main_cls.py
```

Train the MLM demo:

```bash
uv run python main_mlm.py
```

Generate static embedding plots:

```bash
uv run python embedding_viz.py
```

Explore embeddings interactively:

```bash
uv run marimo edit embedding_explorer.py
```

## Notes

The classifier is the more natural task for this dataset: each chunk comes from
either the reptile or bird half of the book, so the `[cls]` representation has a
real document-level signal to learn.

The MLM script is kept as a useful mechanics demo, but the corpus is small and
the word-level vocabulary has a long tail of rare tokens. In practice it mostly
learns common-token frequencies unless the dataset or tokenization strategy is
changed.

Local checkpoints, plots, logs, virtual environments, and notebook session files
are ignored by git. Re-run the scripts above to regenerate them. Early stopping
was not in scope as part of the initial training as classifier accuracy is largely
irrelevant; this is an exploration and demonstration of mechanics rather than
producing an accurate "bird vs reptile" text classifier
