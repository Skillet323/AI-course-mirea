"""Lightweight trained classifier for action-item sentence detection.

The module is designed to be robust in mixed environments:
- it tries to load a cached pickle artifact first,
- if the artifact is missing or incompatible, it retrains from `data/gold`,
- if training also fails, it safely falls back to heuristic scoring.

The classifier is intentionally simple: a binary Logistic Regression model over
combined word- and character-level TF-IDF features.
"""

from __future__ import annotations

import json
import logging
import pickle
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_GOLD_DIR = PROJECT_ROOT / "data" / "gold"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "task_sentence_classifier.pkl"

ACTION_MARKERS = {
    "should",
    "must",
    "need to",
    "needs to",
    "please",
    "action item",
    "action point",
    "follow up",
    "follow-up",
    "will",
    "review",
    "prepare",
    "send",
    "schedule",
    "arrange",
    "contact",
    "coordinate",
    "update",
    "work on",
    "design",
    "develop",
    "implement",
    "create",
    "write",
    "check",
    "look into",
    "set up",
    "take care of",
    "make sure",
    "ensure",
    "need to",
    "нужно",
    "надо",
    "подготовить",
    "сделать",
    "проверить",
    "отправить",
    "согласовать",
    "обновить",
    "выполнить",
    "завершить",
    "доработать",
    "организовать",
    "написать",
    "разработать",
}

_MODEL_CACHE: Any = None
_MODEL_VERSION = 1


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_text_into_sentences(text: str) -> list[str]:
    text = text or ""
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    out: list[str] = []
    for part in parts:
        s = _normalize_space(part)
        if s:
            out.append(s)
    return out


def _looks_like_action_sentence(sentence: str) -> bool:
    s = _normalize_text(sentence)
    if not s:
        return False
    if len(s.split()) < 4:
        return False
    return any(marker in s for marker in ACTION_MARKERS)


def _iter_gold_records() -> Iterable[dict[str, Any]]:
    if not DATA_GOLD_DIR.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(DATA_GOLD_DIR.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("[TASK-CLS] Failed to read %s: %s", path, exc)
    return records


def _meeting_sentences(record: dict[str, Any]) -> list[str]:
    transcript = str(record.get("speaker_transcript") or record.get("transcript") or "")
    sentences = _split_text_into_sentences(transcript)
    # keep speaker-labelled lines as extra candidates
    for line in str(record.get("speaker_transcript") or "").splitlines():
        line = _normalize_space(line)
        if line and line not in sentences:
            sentences.append(line)
    return sentences


def _task_descriptions(record: dict[str, Any]) -> set[str]:
    tasks = record.get("tasks") or []
    descriptions: set[str] = set()
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            desc = _normalize_space(task.get("description") or task.get("task") or "")
            if desc:
                descriptions.add(_normalize_text(desc))
    return descriptions


def _build_training_data() -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []

    records = list(_iter_gold_records())
    if not records:
        return texts, labels

    rng = random.Random(42)
    for record in records:
        task_descs = _task_descriptions(record)
        sentences = _meeting_sentences(record)

        positives: list[str] = []
        negatives: list[str] = []

        for task_desc in task_descs:
            if task_desc:
                positives.append(task_desc)

        for sentence in sentences:
            norm = _normalize_text(sentence)
            if not norm:
                continue
            if norm in task_descs:
                positives.append(sentence)
            elif _looks_like_action_sentence(sentence):
                negatives.append(sentence)
            else:
                negatives.append(sentence)

        # Keep positives unique and clean.
        seen_pos: set[str] = set()
        for item in positives:
            key = _normalize_text(item)[:180]
            if key and key not in seen_pos:
                seen_pos.add(key)
                texts.append(item)
                labels.append(1)

        # Sample negatives to avoid extreme imbalance.
        seen_neg: set[str] = set()
        neg_pool: list[str] = []
        for item in negatives:
            key = _normalize_text(item)[:180]
            if key and key not in seen_neg and len(item.split()) >= 4:
                seen_neg.add(key)
                neg_pool.append(item)

        if positives and neg_pool:
            max_negs = min(len(neg_pool), max(2 * len(positives), 20))
            for item in rng.sample(neg_pool, k=max_negs):
                texts.append(item)
                labels.append(0)
        elif neg_pool:
            for item in neg_pool[:20]:
                texts.append(item)
                labels.append(0)

    return texts, labels


def _build_pipeline():
    from sklearn.compose import FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        (
                            "word",
                            TfidfVectorizer(
                                lowercase=True,
                                ngram_range=(1, 2),
                                min_df=1,
                                max_features=12000,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=1,
                                max_features=14000,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def train_and_save_model(artifact_path: Path = ARTIFACT_PATH) -> Any:
    """Train the classifier from gold data and persist it to disk."""
    texts, labels = _build_training_data()
    if not texts or len(set(labels)) < 2:
        raise RuntimeError("Not enough training data to build task classifier")

    model = _build_pipeline()
    model.fit(texts, labels)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": _MODEL_VERSION,
        "sklearn_model": model,
    }
    with artifact_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("[TASK-CLS] Trained and saved classifier to %s (%d samples)", artifact_path, len(texts))
    return payload


def _load_from_disk() -> Any | None:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if not ARTIFACT_PATH.exists():
        return None

    try:
        with ARTIFACT_PATH.open("rb") as f:
            payload = pickle.load(f)

        model = payload
        if isinstance(payload, dict) and "sklearn_model" in payload:
            if payload.get("model_version") != _MODEL_VERSION:
                raise ValueError("classifier artifact version mismatch")
            model = payload["sklearn_model"]

        _MODEL_CACHE = model
        logger.info("[TASK-CLS] Loaded task classifier from %s", ARTIFACT_PATH)
        return _MODEL_CACHE
    except Exception as exc:
        logger.warning("[TASK-CLS] Failed to load classifier artifact: %s", exc)
        _MODEL_CACHE = None
        return None


def load_model(*, auto_train: bool = True) -> Any | None:
    model = _load_from_disk()
    if model is not None:
        return model

    if not auto_train:
        return None

    try:
        payload = train_and_save_model()
        return payload["sklearn_model"] if isinstance(payload, dict) else payload
    except Exception as exc:
        logger.warning("[TASK-CLS] Auto-training failed: %s", exc)
        return None


def _predict_proba(model: Any, sentences: Sequence[str]) -> list[float]:
    if not sentences:
        return []

    if model is None:
        return []

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(list(sentences))
            # binary classifier: use class 1 if available, otherwise the last column
            if getattr(proba, "ndim", 0) == 2 and proba.shape[1] >= 2:
                idx = 1
                if hasattr(model, "classes_"):
                    classes = list(getattr(model, "classes_"))
                    if 1 in classes:
                        idx = classes.index(1)
                    elif True in classes:
                        idx = classes.index(True)
                return [float(row[idx]) for row in proba]
            return [float(row[-1]) for row in proba]

        if hasattr(model, "decision_function"):
            import numpy as np

            scores = model.decision_function(list(sentences))
            scores = np.asarray(scores, dtype=float)
            scores = 1.0 / (1.0 + np.exp(-scores))
            return [float(x) for x in scores]

        preds = model.predict(list(sentences))
        return [float(x) for x in preds]
    except Exception as exc:
        logger.warning("[TASK-CLS] Prediction failed: %s", exc)
        return []


def predict_sentence_scores(sentences: List[str]) -> List[float]:
    model = load_model(auto_train=True)
    return _predict_proba(model, sentences)


def predict_candidates(
    transcript: str,
    *,
    threshold: float = 0.55,
    max_items: int = 10,
) -> List[Dict[str, Any]]:
    sentences = _split_text_into_sentences(transcript)
    if not sentences:
        return []

    scores = predict_sentence_scores(sentences)
    if not scores:
        return []

    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for sentence, score in sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True):
        norm = _normalize_text(sentence)
        if score < threshold:
            continue
        if len(norm.split()) < 4 and not _looks_like_action_sentence(sentence):
            continue
        key = norm[:160]
        if key in seen:
            continue
        seen.add(key)

        candidates.append(
            {
                "description": sentence[:500],
                "assignee_hint": None,
                "deadline_hint": None,
                "source": "trained_classifier",
                "score": round(float(score), 4),
            }
        )
        if len(candidates) >= max_items:
            break

    return candidates


__all__ = [
    "ARTIFACT_PATH",
    "load_model",
    "predict_candidates",
    "predict_sentence_scores",
    "train_and_save_model",
]
