import re
from pathlib import Path

OUTPUT_DIR = Path("data")


def is_footnote(text):
    return text.startswith("    ")


def is_chapter_header(text):
    return "CHAPTER" in text


def is_illustration(text):
    return "[Illustration" in text


def is_non_english_synonym(text):
    return "SYNONYM" in text and "ENGLISH" not in text


def is_english_synonym(text):
    return "SYNONYM" in text and "ENGLISH" in text


def character_mapping(text):
    translation = str.maketrans({
        "Æ": "AE",
        "æ": "ae",
        "ë": "e",
    })
    return text.translate(translation)


def normalize_emphasis(text):
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"\+([^+]+)\+", r"\1", text)
    return text


def normalize_paragraph(text):
    text = text.lstrip(" ")
    text = text.split("{")[0]
    text = re.sub(r"Fig\. \d{1,3}", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = text.replace("()", "")
    text = normalize_emphasis(text)
    text = character_mapping(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def extract_taxonomies(text):
    taxonomy_start = re.compile(
        r"(?m)^\ufeff?(AMPHIBIA--.*|AVES--BIRDS\.)\s*$"
    )
    chapter_header = re.compile(r"(?m)^CHAPTER I\.\s*$")
    kept_chunks, taxonomies = [], []
    current_pos = 0

    for match in taxonomy_start.finditer(text):
        chapter_match = chapter_header.search(text, match.end())
        if chapter_match is None:
            continue

        kept_chunks.append(text[current_pos:match.start()])
        taxonomy_text = text[match.start():chapter_match.start()].strip()
        taxonomy_name = match.group(1).strip()
        taxonomies.append({
            "name": taxonomy_name,
            "text": taxonomy_text,
        })
        current_pos = chapter_match.start()

    kept_chunks.append(text[current_pos:])
    return "".join(kept_chunks), taxonomies


def extract_synonyms(paragraphs):
    kept_paragraphs, synonyms = [], []
    for i, paragraph in enumerate(paragraphs):
        if is_english_synonym(paragraph) and i > 0:
            synonyms.append((paragraphs[i - 1], paragraph))
        elif not is_english_synonym(paragraph):
            kept_paragraphs.append(paragraph)
    return kept_paragraphs, synonyms


def split_domains(text):
    split_marker = "BIRDS.\n\nINTRODUCTORY CHAPTER.\n"
    reptiles_content, birds_content = text.split(split_marker, maxsplit=1)
    return reptiles_content, birds_content


def clean_paragraphs(text):
    text, taxonomies = extract_taxonomies(text)
    paragraphs = text.split("\n\n")
    paragraphs, synonyms = extract_synonyms(paragraphs)

    cleaned = []
    for paragraph in paragraphs:
        if is_footnote(paragraph):
            continue
        if is_chapter_header(paragraph):
            continue
        if is_illustration(paragraph):
            continue
        if is_non_english_synonym(paragraph):
            continue

        paragraph = normalize_paragraph(paragraph)
        if paragraph:
            cleaned.append(paragraph)

    return cleaned, synonyms, taxonomies


def write_paragraphs(path, paragraphs):
    path.write_text("\n\n".join(paragraphs) + "\n")


def write_synonyms(path, synonyms):
    rows = []
    for header, synonym in synonyms:
        rows.append(header.strip())
        rows.append(synonym.strip())
        rows.append("")
    path.write_text("\n".join(rows))


def write_taxonomies(path, taxonomies):
    rows = []
    for taxonomy in taxonomies:
        rows.append(taxonomy["name"])
        rows.append(taxonomy["text"])
        rows.append("")
    path.write_text("\n".join(rows))


def save_outputs(raw_content):
    OUTPUT_DIR.mkdir(exist_ok=True)
    reptiles_content, birds_content = split_domains(raw_content)

    reptiles_paragraphs, reptiles_synonyms, reptiles_taxonomies = clean_paragraphs(reptiles_content)
    birds_paragraphs, birds_synonyms, birds_taxonomies = clean_paragraphs(birds_content)
    all_paragraphs = reptiles_paragraphs + birds_paragraphs
    all_synonyms = reptiles_synonyms + birds_synonyms
    all_taxonomies = reptiles_taxonomies + birds_taxonomies

    write_paragraphs(OUTPUT_DIR / "reptiles_paragraphs.txt", reptiles_paragraphs)
    write_paragraphs(OUTPUT_DIR / "birds_paragraphs.txt", birds_paragraphs)
    write_paragraphs(OUTPUT_DIR / "paragraphs.txt", all_paragraphs)
    write_synonyms(OUTPUT_DIR / "synonyms.txt", all_synonyms)
    write_taxonomies(OUTPUT_DIR / "taxonomies.txt", all_taxonomies)

    return {
        "reptile_paragraphs": len(reptiles_paragraphs),
        "bird_paragraphs": len(birds_paragraphs),
        "synonyms": len(all_synonyms),
        "taxonomies": len(all_taxonomies),
    }


if __name__ == "__main__":
    raw_content = Path("birds_reptiles_prepped.txt").read_text()
    stats = save_outputs(raw_content)
    for name, value in stats.items():
        print(f"{name}: {value}")
