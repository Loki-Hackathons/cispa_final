"""Data loading for the Text Watermark Localization dataset.

The dataset (`SprintML/watermark_localization`) has three splits: train (90, labeled),
validation (90, labeled), test (1320, unlabeled). Records are index-aligned:
``token_ids``, ``token_pieces`` and (train/val) ``labels`` all have the same length.

``token_ids`` are authoritative — we never retokenize ``text``.

Run a quick check:  ``python -m src.load_data --check``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import config


@dataclass
class Document:
    document_id: str
    token_ids: list[int]
    token_pieces: list[str]
    text: str
    labels: Optional[list[int]] = None  # None for test

    def __post_init__(self) -> None:
        n = len(self.token_ids)
        if len(self.token_pieces) != n:
            raise ValueError(
                f"{self.document_id}: token_pieces ({len(self.token_pieces)}) "
                f"!= token_ids ({n})"
            )
        if self.labels is not None and len(self.labels) != n:
            raise ValueError(
                f"{self.document_id}: labels ({len(self.labels)}) != token_ids ({n})"
            )

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


def _row_to_document(row: dict) -> Document:
    labels = row.get("labels", None)
    return Document(
        document_id=str(row["document_id"]),
        token_ids=list(row["token_ids"]),
        token_pieces=list(row["token_pieces"]),
        text=row.get("text", ""),
        labels=list(labels) if labels is not None else None,
    )


# Map split name -> file name in the dataset directory.
_SPLIT_FILE = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}


def _read_jsonl(path: str) -> list[Document]:
    docs: list[Document] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(_row_to_document(json.loads(line)))
    return docs


def load_split(split: str) -> list[Document]:
    """Load one split ('train' | 'validation' | 'test') as a list of Document.

    Primary source (like the rest of the team): the local ``{split}.jsonl`` file inside
    ``config.DATASET_DIR`` (HF dataset repo cloned to scratch / ``data/``). If that file is
    absent, fall back to the HuggingFace ``datasets`` library (``config.DATASET_ID``).
    """
    alias = {"val": "validation", "valid": "validation", "dev": "validation"}
    split = alias.get(split, split)

    # 1) Local .jsonl files (train.jsonl / validation.jsonl / test.jsonl).
    fname = _SPLIT_FILE.get(split, f"{split}.jsonl")
    local = Path(config.DATASET_DIR) / fname
    if local.is_file():
        return _read_jsonl(str(local))

    # 2) Fallback: HuggingFace hub / local dataset dir understood by `datasets`.
    from datasets import load_dataset  # local import: heavy optional dep

    config.ensure_dirs()
    ds = load_dataset(config.DATASET_ID, split=split, cache_dir=str(config.CACHE_DIR))
    return [_row_to_document(row) for row in ds]


def load_all() -> dict[str, list[Document]]:
    out: dict[str, list[Document]] = {}
    for split in ("train", "validation", "test"):
        try:
            out[split] = load_split(split)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[load_data] could not load split '{split}': {exc}")
    return out


def _check() -> None:
    data = load_all()
    for split, docs in data.items():
        n_docs = len(docs)
        n_tokens = sum(d.n_tokens for d in docs)
        has_labels = any(d.labels is not None for d in docs)
        print(f"{split:11s} docs={n_docs:5d} tokens={n_tokens:8d} labeled={has_labels}")
        if docs:
            d = docs[0]
            print(f"  sample {d.document_id}: n={d.n_tokens} "
                  f"pieces[:5]={d.token_pieces[:5]} "
                  f"labels[:5]={None if d.labels is None else d.labels[:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the watermark dataset.")
    parser.add_argument("--check", action="store_true", help="Print split stats.")
    args = parser.parse_args()
    if args.check:
        _check()
    else:
        parser.print_help()
