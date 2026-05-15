
"""Compare the rule-based fallback against the trained sentence classifier.

This module is intentionally dependency-light so it can be imported in tests
and used as a simple CLI helper.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.services.task_classifier import predict_candidates
from src.services.task_extraction import extract_tasks_rule_based

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = PROJECT_ROOT / "data" / "gold"


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text or "")
    return [re.sub(r"\s+", " ", p or "").strip() for p in parts if re.sub(r"\s+", " ", p or "").strip()]


def sim(a: str, b: str) -> float:
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


def parse_gold_dir(gold_dir: Path = GOLD_DIR) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fp in sorted(gold_dir.glob("*.json")):
        with fp.open("r", encoding="utf-8") as f:
            item = json.load(f)
        if isinstance(item, dict) and item.get("transcript") and item.get("tasks"):
            item["_source_file"] = fp.name
            records.append(item)
    return records


def evaluate_tasks(pred_tasks: list[dict[str, Any]], gold_tasks: list[dict[str, Any]], threshold: float = 0.28) -> dict[str, Any]:
    pred_tasks = pred_tasks or []
    gold_tasks = gold_tasks or []

    matches: dict[int, int] = {}
    used_gold: set[int] = set()
    gold_descs = [normalize_text(str(g.get("description") or g.get("task") or "")) for g in gold_tasks]

    for i, pred in enumerate(pred_tasks):
        pdesc = normalize_text(str(pred.get("description") or pred.get("task") or ""))
        best_idx, best_score = -1, 0.0
        for j, gdesc in enumerate(gold_descs):
            if j in used_gold:
                continue
            score = sim(pdesc, gdesc)
            if score > best_score:
                best_idx, best_score = j, score
        if best_idx >= 0 and best_score >= threshold:
            matches[i] = best_idx
            used_gold.add(best_idx)

    tp = len(matches)
    fp = max(0, len(pred_tasks) - tp)
    fn = max(0, len(gold_tasks) - tp)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted": len(pred_tasks),
        "gold": len(gold_tasks),
        "matched": tp,
    }


def compare_models(gold_dir: Path = GOLD_DIR) -> dict[str, Any]:
    records = parse_gold_dir(gold_dir)
    baseline_scores = []
    trained_scores = []

    for rec in records:
        transcript = rec.get("transcript") or ""
        gold_tasks = rec.get("tasks") or []

        baseline = extract_tasks_rule_based(transcript)
        trained = predict_candidates(transcript, threshold=0.70, max_items=10)

        baseline_scores.append(evaluate_tasks(baseline, gold_tasks))
        trained_scores.append(evaluate_tasks(trained, gold_tasks))

    def avg(key: str, rows: list[dict[str, Any]]) -> float:
        vals = [float(r.get(key, 0.0) or 0.0) for r in rows]
        return sum(vals) / max(1, len(vals))

    return {
        "meetings": len(records),
        "rule_based": {
            "precision": avg("precision", baseline_scores),
            "recall": avg("recall", baseline_scores),
            "f1": avg("f1", baseline_scores),
        },
        "trained_classifier": {
            "precision": avg("precision", trained_scores),
            "recall": avg("recall", trained_scores),
            "f1": avg("f1", trained_scores),
        },
    }


def main() -> None:
    print(json.dumps(compare_models(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
