
"""Train a lightweight action-item sentence classifier from the gold corpus.

Usage:
    python -m src.training.train_task_classifier
"""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "task_sentence_classifier.pkl"


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text or "")
    out: list[str] = []
    for part in parts:
        s = re.sub(r"\s+", " ", part or "").strip()
        if s:
            out.append(s)
    return out


def task_similarity(a: str, b: str) -> float:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    seq = SequenceMatcher(None, a_n, b_n).ratio()
    a_tokens = set(a_n.split())
    b_tokens = set(b_n.split())
    if not a_tokens or not b_tokens:
        return seq
    jacc = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    return 0.6 * jacc + 0.4 * seq


def load_gold_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fp in sorted(GOLD_DIR.glob("*.json")):
        with fp.open("r", encoding="utf-8") as f:
            item = json.load(f)
        if isinstance(item, dict) and item.get("transcript") and item.get("tasks"):
            item["_source_file"] = fp.name
            records.append(item)
    return records


def build_dataset(records: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    X: list[str] = []
    y: list[int] = []

    for rec in records:
        transcript = rec.get("transcript") or ""
        gold_tasks = rec.get("tasks") or []
        gold_descs = [
            str(t.get("description") or t.get("task") or "").strip()
            for t in gold_tasks
            if isinstance(t, dict)
        ]
        gold_descs = [d for d in gold_descs if d]

        for sent in split_sentences(transcript):
            norm = normalize_text(sent)
            if len(norm.split()) < 4:
                continue
            score = max((task_similarity(sent, gd) for gd in gold_descs), default=0.0)
            # Confident labels only: positives and obvious negatives.
            if score >= 0.34:
                X.append(sent)
                y.append(1)
            elif score <= 0.16:
                X.append(sent)
                y.append(0)

    return X, y


def train() -> dict[str, Any]:
    records = load_gold_records()
    X, y = build_dataset(records)
    if not X:
        raise RuntimeError("No training data could be built from data/gold")

    print(f"Loaded {len(records)} gold meetings")
    print(f"Training examples: {len(X)} (positives={sum(y)}, negatives={len(y) - sum(y)})")

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=40000,
                    lowercase=True,
                    strip_accents=None,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )

    pipeline.fit(X, y)

    # Simple in-sample metrics for sanity.
    preds = pipeline.predict(X)
    report = classification_report(y, preds, output_dict=True, zero_division=0)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("wb") as f:
        pickle.dump(pipeline, f)

    summary = {
        "artifact_path": str(ARTIFACT_PATH),
        "training_examples": len(X),
        "positive_examples": int(sum(y)),
        "negative_examples": int(len(y) - sum(y)),
        "precision_pos": report.get("1", {}).get("precision", 0.0),
        "recall_pos": report.get("1", {}).get("recall", 0.0),
        "f1_pos": report.get("1", {}).get("f1-score", 0.0),
    }
    return summary


def main() -> None:
    summary = train()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
