import json
import re
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = ["[pad]", "[unk]", "[eos]", "[mask]", "[cls]"]
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]")
DATA_DIR = Path("data")
MAX_SEQ_LEN = 128


def read_paragraphs(path):
    text = path.read_text()
    return [paragraph for paragraph in text.split("\n\n") if paragraph.strip()]


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


def build_vocabulary(paragraphs, min_count=1):
    counts = Counter()
    for paragraph in paragraphs:
        counts.update(tokenize(paragraph))

    vocabulary = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    for token, count in counts.most_common():
        if count >= min_count and token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_paragraph(paragraph, vocabulary):
    tokens = ["[cls]"] + tokenize(paragraph) + ["[eos]"]
    return [vocabulary.get(token, vocabulary["[unk]"]) for token in tokens]


def encode_paragraphs(paragraphs, vocabulary):
    return [encode_paragraph(paragraph, vocabulary) for paragraph in paragraphs]


def make_chunks(encoded_paragraphs, vocabulary, max_seq_len=MAX_SEQ_LEN):
    pad_id = vocabulary["[pad]"]
    chunks = []
    for paragraph in encoded_paragraphs:
        for start in range(0, len(paragraph), max_seq_len):
            chunk = paragraph[start:start + max_seq_len]
            if len(chunk) < max_seq_len:
                chunk = chunk + [pad_id] * (max_seq_len - len(chunk))
            chunks.append(chunk)
    return chunks


def write_vocabulary(path, vocabulary):
    rows = sorted(vocabulary.items(), key=lambda item: item[1])
    path.write_text("\n".join(f"{token}\t{token_id}" for token, token_id in rows) + "\n")


def write_token_ids(path, encoded_paragraphs):
    rows = [" ".join(str(token_id) for token_id in paragraph) for paragraph in encoded_paragraphs]
    path.write_text("\n".join(rows) + "\n")


def write_labels(path, labels):
    path.write_text("\n".join(labels) + "\n")


def save_outputs():
    reptiles_paragraphs = read_paragraphs(DATA_DIR / "reptiles_paragraphs.txt")
    birds_paragraphs = read_paragraphs(DATA_DIR / "birds_paragraphs.txt")
    all_paragraphs = reptiles_paragraphs + birds_paragraphs

    vocabulary = build_vocabulary(all_paragraphs)
    encoded_reptiles = encode_paragraphs(reptiles_paragraphs, vocabulary)
    encoded_birds = encode_paragraphs(birds_paragraphs, vocabulary)
    encoded_paragraphs = encoded_reptiles + encoded_birds
    reptile_chunks = make_chunks(encoded_reptiles, vocabulary)
    bird_chunks = make_chunks(encoded_birds, vocabulary)
    all_chunks = reptile_chunks + bird_chunks

    write_vocabulary(DATA_DIR / "vocabulary.txt", vocabulary)
    write_token_ids(DATA_DIR / "reptiles_token_ids.txt", encoded_reptiles)
    write_token_ids(DATA_DIR / "birds_token_ids.txt", encoded_birds)
    write_token_ids(DATA_DIR / "token_ids.txt", encoded_paragraphs)
    write_token_ids(DATA_DIR / "chunks.txt", all_chunks)
    write_labels(
        DATA_DIR / "chunk_labels.txt",
        ["reptile"] * len(reptile_chunks) + ["bird"] * len(bird_chunks),
    )
    (DATA_DIR / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2) + "\n")

    return {
        "vocab_size": len(vocabulary),
        "encoded_paragraphs": len(encoded_paragraphs),
        "chunks": len(all_chunks),
    }


if __name__ == "__main__":
    stats = save_outputs()
    for name, value in stats.items():
        print(f"{name}: {value}")
