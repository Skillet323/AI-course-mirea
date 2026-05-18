"""Lightweight action-item sentence classifier.

The module prefers a trained local model, but it can always fall back to a
heuristic scorer so downstream extraction never becomes empty just because the
artifact is missing or incompatible.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion, Pipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "task_sentence_classifier.pkl"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"

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
    "look at",
    "decide",
    "figure out",
    "define",
    "determine",
    "analyze",
    "discuss",
    "finalize",
    "set up",
    "assign",
    "distribute",
    "present",
}

NEGATIVE_MARKERS = {
    "agenda",
    "introduction",
    "introduce ourselves",
    "my name is",
    "hello everybody",
    "good morning",
    "project announcement",
    "meeting agenda",
    "icebreaker",
    "favorite animal",
    "favorite characteristic",
    "design stages",
    "team building",
    "status update",
}

_MODEL_CACHE: Any = None


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_speaker_prefix(text: str) -> tuple[Optional[str], str]:
    m = re.match(
        r"^\s*((?:SPEAKER_\d+)|(?:Speaker\s+\d+)|(?:[A-Z])|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}))\s*:\s*(.+)$",
        text or "",
    )
    if not m:
        return None, _normalize_space(text)
    return _normalize_space(m.group(1)), _normalize_space(m.group(2))


def _split_text_into_sentences(text: str) -> list[dict[str, Any]]:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text or "")
    out: list[dict[str, Any]] = []
    for part in parts:
        raw = _normalize_space(part)
        if not raw:
            continue
        speaker, body = _split_speaker_prefix(raw)
        if not body:
            continue
        out.append({"speaker_hint": speaker, "text": body, "raw": raw})
    return out


def _looks_like_task(text: str) -> bool:
    s = _normalize_text(text)
    if not s or len(s.split()) < 4:
        return False
    if any(marker in s for marker in NEGATIVE_MARKERS):
        return False
    return any(marker in s for marker in ACTION_MARKERS) or bool(
        re.search(r"\b(?:should|must|need to|needs to|will|let us|let's|have to|going to|responsible for|assigned to)\b", s)
    )


def _heuristic_score(text: str) -> float:
    s = _normalize_text(text)
    if not s:
        return 0.0

    score = 0.0
    if any(marker in s for marker in NEGATIVE_MARKERS):
        return 0.0

    markers = sum(1 for marker in ACTION_MARKERS if marker in s)
    score += min(0.45, 0.12 * markers)

    if re.search(r"\b(should|must|need to|needs to|have to|let us|let's|going to|will need|will have)\b", s):
        score += 0.22
    if re.search(r"\b(work on|prepare|write|send|review|check|update|implement|create|design|develop|finalize|decide|define|determine|analyze|figure out)\b", s):
        score += 0.20
    if re.search(r"\b(by|before)\b.+\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next)\b", s):
        score += 0.08
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", text or ""):
        score += 0.05

    length = len(s.split())
    if length < 5:
        score -= 0.10
    elif length > 35:
        score -= 0.05

    return max(0.0, min(1.0, score))


def _read_gold_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not GOLD_DIR.exists():
        return records
    for fp in sorted(GOLD_DIR.glob("*.json")):
        try:
            with fp.open("r", encoding="utf-8") as f:
                item = json.load(f)
        except Exception:
            continue
        if isinstance(item, dict) and item.get("transcript") and item.get("tasks"):
            item["_source_file"] = fp.name
            records.append(item)
    return records


def _task_similarity(a: str, b: str) -> float:
    a_n = _normalize_text(a)
    b_n = _normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    a_tokens = set(a_n.split())
    b_tokens = set(b_n.split())
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    seq = 1.0 - min(1.0, abs(len(a_n) - len(b_n)) / max(len(a_n), len(b_n), 1))
    return 0.7 * overlap + 0.3 * seq


def _build_dataset(records: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    X: list[str] = []
    y: list[int] = []

    for rec in records:
        transcript = rec.get("transcript") or ""
        gold_tasks = rec.get("tasks") or []
        gold_texts = []
        for task in gold_tasks:
            if not isinstance(task, dict):
                continue
            desc = _normalize_space(str(task.get("description") or task.get("task") or ""))
            snippet = _normalize_space(str(task.get("source_snippet") or task.get("evidence") or ""))
            if desc:
                gold_texts.append(desc)
            if snippet:
                gold_texts.append(snippet)

        sentences = _split_text_into_sentences(transcript)
        for item in sentences:
            sent = item["text"]
            norm = _normalize_text(sent)
            if len(norm.split()) < 4:
                continue

            best = max((_task_similarity(sent, gt) for gt in gold_texts), default=0.0)
            heuristic = _heuristic_score(sent)

            if best >= 0.28 or heuristic >= 0.68:
                X.append(sent)
                y.append(1)
            elif best <= 0.12 and heuristic <= 0.18:
                X.append(sent)
                y.append(0)

        # Add the gold tasks and snippets as positively labelled anchors.
        for gt in gold_texts:
            if len(_normalize_text(gt).split()) >= 3:
                X.append(gt)
                y.append(1)

    return X, y


def _train_model() -> Any | None:
    records = _read_gold_records()
    X, y = _build_dataset(records)
    if not X:
        return None

    counts = Counter(y)
    logger.info(
        "[TASK-CLS] Training fresh classifier on %d examples (pos=%d neg=%d)",
        len(X),
        counts.get(1, 0),
        counts.get(0, 0),
    )

    model = Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                min_df=1,
                                max_features=30000,
                                lowercase=True,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=1,
                                max_features=30000,
                                lowercase=True,
                            ),
                        ),
                    ]
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
    model.fit(X, y)

    preds = model.predict(X)
    report = classification_report(y, preds, output_dict=True, zero_division=0)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("wb") as f:
        pickle.dump(
            {
                "model": model,
                "sklearn_version": __import__("sklearn").__version__,
                "training_examples": len(X),
                "positive_examples": int(counts.get(1, 0)),
                "negative_examples": int(counts.get(0, 0)),
                "f1_pos": float(report.get("1", {}).get("f1-score", 0.0)),
            },
            f,
        )

    logger.info("[TASK-CLS] Saved classifier artifact to %s", ARTIFACT_PATH)
    return model


def _unwrap_model(obj: Any) -> Any | None:
    if obj is None:
        return None
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def _is_model_usable(model: Any) -> bool:
    try:
        sample = ["We should prepare the report by Friday."]
        if not hasattr(model, "predict_proba"):
            return False
        proba = model.predict_proba(sample)
        return bool(proba) and len(proba[0]) >= 2
    except Exception:
        return False


def load_model() -> Any | None:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if ARTIFACT_PATH.exists():
        try:
            with ARTIFACT_PATH.open("rb") as f:
                artifact = pickle.load(f)
            candidate = _unwrap_model(artifact)
            if _is_model_usable(candidate):
                _MODEL_CACHE = candidate
                logger.info("[TASK-CLS] Loaded task classifier from %s", ARTIFACT_PATH)
                return _MODEL_CACHE
            logger.warning("[TASK-CLS] Loaded artifact is not usable, retraining")
        except Exception as exc:
            logger.warning("[TASK-CLS] Failed to load classifier artifact: %s", exc)

    _MODEL_CACHE = _train_model()
    return _MODEL_CACHE


def predict_sentence_scores(sentences: List[str]) -> List[float]:
    if not sentences:
        return []

    model = load_model()
    heuristic_scores = [_heuristic_score(sentence) for sentence in sentences]

    if model is None:
        return heuristic_scores

    try:
        proba = model.predict_proba(sentences)
        model_scores = [float(row[1]) if len(row) > 1 else float(row[0]) for row in proba]
        return [max(0.0, min(1.0, 0.70 * ms + 0.30 * hs)) for ms, hs in zip(model_scores, heuristic_scores)]
    except Exception as exc:
        logger.warning("[TASK-CLS] Prediction failed: %s", exc)
        return heuristic_scores


def predict_candidates(
    transcript: str,
    *,
    threshold: float = 0.55,
    max_items: int = 10,
) -> List[Dict[str, Any]]:
    items = _split_text_into_sentences(transcript)
    if not items:
        return []

    texts = [item["text"] for item in items]
    scores = predict_sentence_scores(texts)
    ranked = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)

    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for item, score in ranked:
        text = item["text"]
        norm = _normalize_text(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)

        if any(marker in norm for marker in ("agenda", "hello everybody", "good morning", "meeting agenda", "project announcement", "introduce ourselves", "my name is", "favorite animal", "favourite animal", "icebreaker")):
            continue
        if score < threshold and not _looks_like_task(text):
            continue
        if len(norm.split()) < 4:
            continue

        candidate: Dict[str, Any] = {
            "description": text[:500],
            "assignee_hint": None,
            "deadline_hint": None,
            "speaker_hint": item.get("speaker_hint"),
            "source_snippet": item.get("raw") or text[:120],
            "source": "trained_classifier" if load_model() is not None else "heuristic_classifier",
            "score": round(float(score), 4),
        }
        candidates.append(candidate)
        if len(candidates) >= max_items:
            break

    return candidates
