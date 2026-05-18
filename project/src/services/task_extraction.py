"""Task extraction with OpenRouter / NVIDIA, candidate gating and strict fallback.

The extractor intentionally works in three layers:
1) candidate generation from the transcript (rule-based + trained classifier)
2) LLM normalization over only those candidates
3) strict rule-based fallback when LLM is unavailable or unhelpful

Returned task dicts are normalized to:
    {
        "description": str,
        "assignee_hint": str|None,
        "deadline_hint": str|None,
        "speaker_hint": str|None,
        "source": "openrouter"|"nvidia"|"openrouter_text"|"rule_based"|"trained_classifier"|"heuristic_classifier",
        "source_snippet": str|None,
        "model": str|None,
        "candidate_id": int|None,
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings
from .task_classifier import predict_candidates as predict_trained_candidates

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_NVIDIA_BASE_URL_DEFAULT = "https://integrate.api.nvidia.com/v1"
MAX_OPENROUTER_RETRIES = 2
MAX_NVIDIA_RETRIES = 2
BASE_BACKOFF_SEC = 1.8

SYSTEM_PROMPT = """You extract concrete action items from a meeting transcript.

Rules:
- Use ONLY the provided candidate sentences. Never invent new tasks.
- Keep only concrete work items, requests, decisions, follow-ups, assignments, or explicit next steps.
- Reject agenda items, introductions, meeting narration, project background, and generic topic summaries.
- If a candidate is not an actual action item, omit it.
- Return valid JSON only, preferably as {"tasks": [...]}.

Each task object should contain:
{
  "candidate_id": 12,
  "description": "short normalized task description",
  "assignee_hint": "optional person/role",
  "deadline_hint": "optional deadline",
  "speaker_hint": "optional speaker label",
  "source_snippet": "verbatim candidate sentence or short quote"
}
"""

USER_TEMPLATE = """Meeting ref: {meeting_ref}
Language: {language}
Duration (sec): {duration_sec}
Transcript confidence: {transcript_confidence}

You may only use the following candidate sentences:
{candidate_json}

Transcript excerpt:
{transcript_excerpt}
"""

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "had", "are", "was", "were",
    "will", "would", "could", "should", "need", "needs", "please", "about", "into", "onto", "then",
    "than", "them", "they", "their", "there", "here", "what", "when", "where", "why", "how", "who",
    "whom", "which", "your", "our", "you", "we", "uh", "um", "mm", "yeah", "okay", "ok", "right",
    "just", "also", "still", "very", "really", "maybe",
}

NEGATIVE_PHRASES = (
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
    "favourite animal",
    "favorite characteristic",
    "favourite characteristic",
    "design stages",
    "team building",
    "status update",
    "project brief",
    "what the project is about",
)

ACTION_PHRASES = (
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
    "confirm",
    "complete",
    "deliver",
    "draft",
    "research",
    "test",
    "review",
    "document",
    "publish",
)

MODEL_HINT_PHRASES = (
    "project manager",
    "designer",
    "developer",
    "engineer",
    "marketing",
    "ui",
    "ux",
    "technical",
    "facilitator",
    "moderator",
    "host",
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _preview(text: str, limit: int = 800) -> str:
    text = (text or "").replace("\r", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _split_speaker_prefix(text: str) -> tuple[Optional[str], str]:
    m = re.match(
        r"^\s*((?:SPEAKER_\d+)|(?:Speaker\s+\d+)|(?:[A-Z])|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}))\s*:\s*(.+)$",
        text or "",
    )
    if not m:
        return None, _normalize_space(text)

    speaker = _normalize_space(m.group(1))
    body = _normalize_space(m.group(2))
    return speaker, body


def _tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    return [t for t in text.split() if t and t not in STOPWORDS and len(t) > 2]


def _looks_like_task(sentence: str) -> bool:
    s = normalize_text(sentence)
    if len(s.split()) < 4:
        return False
    if any(phrase in s for phrase in NEGATIVE_PHRASES):
        return False
    if any(phrase in s for phrase in ACTION_PHRASES):
        return True
    return bool(re.search(r"\b(?:should|must|need to|needs to|will|let us|let's|have to|going to|responsible for|assigned to|please)\b", s))


def _heuristic_score(text: str) -> float:
    s = normalize_text(text)
    if not s:
        return 0.0
    if any(phrase in s for phrase in NEGATIVE_PHRASES):
        return 0.0

    score = 0.0
    marker_hits = sum(1 for phrase in ACTION_PHRASES if phrase in s)
    score += min(0.45, 0.10 * marker_hits)

    if re.search(r"\b(should|must|need to|needs to|have to|let us|let's|going to|will need|will have)\b", s):
        score += 0.22
    if re.search(r"\b(work on|prepare|write|send|review|check|update|implement|create|design|develop|finalize|decide|define|determine|analyze|figure out|complete|deliver|draft|research|test)\b", s):
        score += 0.20
    if re.search(r"\b(by|before)\b.+\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next)\b", s):
        score += 0.08
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", text or ""):
        score += 0.04

    length = len(s.split())
    if length < 5:
        score -= 0.10
    elif length > 35:
        score -= 0.05

    return max(0.0, min(1.0, score))


def _task_supported_by_transcript(description: str, transcript: str, min_overlap: float = 0.10) -> bool:
    desc_tokens = set(_tokenize(description))
    tr_tokens = set(_tokenize(transcript))

    if len(desc_tokens) < 3 or not tr_tokens:
        return False

    overlap = len(desc_tokens & tr_tokens) / max(1, len(desc_tokens))
    if overlap >= min_overlap:
        return True

    if len(desc_tokens & tr_tokens) >= 2:
        return True

    return False


def _guess_deadline(sentence: str) -> Optional[str]:
    m = re.search(
        r"(?:by|до|к|к\s+)"
        r"(\d{1,2}(?:[./-]\d{1,2})?(?:[./-]\d{2,4})?|"
        r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"понедельник|вторник|среда|четверг|пятница|суббота|воскресенье))",
        sentence,
        flags=re.IGNORECASE,
    )
    return _normalize_space(m.group(1)) if m else None


_ASSIGNEE_DEADLINE_TOKENS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье",
    "today", "tomorrow", "tonight", "next", "week", "month",
}


def _guess_assignee(sentence: str) -> Optional[str]:
    patterns = [
        r"(?:for|to|assigned to)\s+([A-Za-zА-ЯЁа-яё][\w.'-]*(?:\s+[A-Za-zА-ЯЁа-яё][\w.'-]*){0,2})",
        r"(?:для|от)\s+([A-Za-zА-ЯЁа-яё][\w.'-]*(?:\s+[A-Za-zА-ЯЁа-яё][\w.'-]*){0,2})",
        r"\bby\s+([A-Za-zА-ЯЁа-яё][\w.'-]*(?:\s+[A-Za-zА-ЯЁа-яё][\w.'-]*){0,2})",
        r"@([\w.-]+)",
    ]
    blocked = {
        "review", "prepare", "send", "schedule", "arrange", "contact", "coordinate", "update", "work", "design",
        "develop", "implement", "create", "write", "check", "look", "figure", "define", "determine", "analyze",
        "discuss", "finalize", "complete", "deliver", "draft", "research", "test", "help", "team", "the", "a",
        "an", "and", "or", "to", "for", "by", "of", "with", "from", "on", "in", "it", "this", "that",
    }
    for pattern in patterns:
        m = re.search(pattern, sentence, flags=re.IGNORECASE)
        if not m:
            continue
        candidate = _normalize_space(m.group(1))
        low = normalize_text(candidate)
        if not candidate or low in _ASSIGNEE_DEADLINE_TOKENS:
            continue
        if low.split() and low.split()[0] in blocked:
            continue
        if not re.fullmatch(r"[A-ZА-ЯЁ][a-zа-яё'\-]*(?:\s+[A-ZА-ЯЁ][a-zа-яё'\-]*){0,2}", candidate):
            continue
        return candidate
    return None


def _is_meta_task(text: str) -> bool:
    s = normalize_text(text)
    bad_phrases = [
        "review and summarize action items",
        "extract action items from the meeting transcript",
        "meeting transcript",
        "summarize action items",
        "introduce participants",
        "confirm meeting agenda",
        "project goal and objectives",
        "outline the project structure",
        "describe the functional design process",
        "explain the tool training exercise",
        "introduce the user interface design approach",
        "confirm attendance",
        "review current remote control features",
        "start the meeting",
        "confirm everyone is ready",
        "read the entire transcript first",
        "meeting id",
        "language",
        "duration",
    ]
    return any(p in s for p in bad_phrases)


def _normalize_assignee_hint(hint: Any) -> Optional[str]:
    if hint is None:
        return None

    text = _normalize_space(str(hint))
    if not text:
        return None

    lower = normalize_text(text)
    generic_markers = [
        "assign tasks",
        "ensure",
        "review",
        "summarize",
        "prepare",
        "confirm",
        "lead",
        "team",
        "members",
        "based on meeting content",
        "meeting content",
        "project manager",
        "facilitator",
        "design team",
        "marketing expert",
        "product designer",
        "assign specific tasks",
        "should do",
        "task owner",
    ]

    if len(text.split()) > 5:
        return None

    if any(marker in lower for marker in generic_markers):
        if not re.fullmatch(r"[A-Za-zА-ЯЁ][\w.-]+(?:\s+[A-Za-zА-ЯЁ][\w.-]+)?", text):
            return None

    return text


def _normalize_deadline_hint(hint: Any) -> Optional[str]:
    if hint is None:
        return None
    text = _normalize_space(str(hint))
    return text or None


def _sentence_items(transcript: str) -> list[dict[str, Any]]:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", transcript or "")
    items: list[dict[str, Any]] = []
    for part in parts:
        raw = _normalize_space(part)
        if not raw:
            continue
        speaker, body = _split_speaker_prefix(raw)
        if len(body) < 12:
            continue
        items.append({"speaker_hint": speaker, "text": body, "raw": raw})
    return items


def _candidate_pool(transcript: str, *, limit: int = 24) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_candidate(text: str, *, speaker_hint: Optional[str], source: str, score: float, source_snippet: Optional[str] = None) -> None:
        desc = _normalize_space(text)
        if not desc or _is_meta_task(desc):
            return
        key = normalize_text(desc)[:150]
        if not key or key in seen:
            return
        seen.add(key)
        items.append(
            {
                "id": len(items) + 1,
                "description": desc[:500],
                "speaker_hint": speaker_hint,
                "source_snippet": _normalize_space(source_snippet or desc)[:180],
                "source": source,
                "score": round(float(score), 4),
                "assignee_hint": _guess_assignee(desc),
                "deadline_hint": _guess_deadline(desc),
            }
        )

    # Rule-based candidates.
    for item in _sentence_items(transcript):
        text = item["text"]
        if _looks_like_task(text):
            score = 0.58 + min(0.25, 0.04 * len(_tokenize(text)))
            add_candidate(text, speaker_hint=item.get("speaker_hint"), source="rule_hint", score=score, source_snippet=item.get("raw"))

    # Trained / heuristic classifier candidates.
    try:
        classifier_candidates = predict_trained_candidates(transcript, threshold=0.46, max_items=20)
    except Exception as exc:
        logger.warning("[TASK] classifier candidate generation failed: %s", exc)
        classifier_candidates = []

    for cand in classifier_candidates:
        add_candidate(
            cand.get("description") or "",
            speaker_hint=cand.get("speaker_hint"),
            source=str(cand.get("source") or "trained_classifier"),
            score=float(cand.get("score") or 0.0) + 0.02,
            source_snippet=cand.get("source_snippet") or cand.get("description"),
        )

    # If we still have very few candidates, keep a few strong transcript lines.
    if len(items) < 6:
        ranked = sorted(_sentence_items(transcript), key=lambda x: _heuristic_score(x["text"]), reverse=True)
        for item in ranked:
            if len(items) >= limit:
                break
            score = _heuristic_score(item["text"])
            if score < 0.35:
                continue
            add_candidate(item["text"], speaker_hint=item.get("speaker_hint"), source="heuristic_classifier", score=score, source_snippet=item.get("raw"))

    items.sort(key=lambda x: (x.get("score", 0.0), len(x.get("description", ""))), reverse=True)
    return items[:limit]


def _candidate_map(pool: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item["id"]): item for item in pool if item.get("id") is not None}


def _candidate_similarity(a: str, b: str) -> float:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    a_tokens = set(a_n.split())
    b_tokens = set(b_n.split())
    if not a_tokens or not b_tokens:
        return SequenceMatcher(None, a_n, b_n).ratio()
    overlap = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    seq = SequenceMatcher(None, a_n, b_n).ratio()
    return 0.65 * overlap + 0.35 * seq


def _add_task(tasks: List[Dict[str, Any]], item: Dict[str, Any], transcript: str) -> None:
    desc = _normalize_space(str(item.get("description") or item.get("task") or ""))
    if not desc or _is_meta_task(desc):
        return
    if not _task_supported_by_transcript(desc, transcript):
        return

    speaker_hint = item.get("speaker_hint") or None
    speaker_hint = _normalize_space(str(speaker_hint)) if speaker_hint else None

    assignee_hint = _normalize_assignee_hint(item.get("assignee_hint") or item.get("assignee"))
    deadline_hint = _normalize_deadline_hint(item.get("deadline_hint") or item.get("deadline"))

    source_snippet = item.get("source_snippet") or item.get("evidence")
    source_snippet = _normalize_space(str(source_snippet)) if source_snippet else None

    out: Dict[str, Any] = {
        "description": desc[:500],
        "assignee_hint": assignee_hint,
        "deadline_hint": deadline_hint,
        "source": item.get("source") or "openrouter",
    }
    if speaker_hint:
        out["speaker_hint"] = speaker_hint
    if source_snippet:
        out["source_snippet"] = source_snippet[:180]
    if item.get("model"):
        out["model"] = item["model"]
    if item.get("candidate_id") is not None:
        out["candidate_id"] = item["candidate_id"]

    tasks.append(out)


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def _extract_tasks_simple(transcript: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in _sentence_items(transcript):
        sentence = item["text"]
        if not _looks_like_task(sentence):
            continue

        sentence = re.sub(
            r"^(?:ну|okay|ok|please|let's|we should|we need to|нужно|надо)\s*,?\s*",
            "",
            sentence,
            flags=re.IGNORECASE,
        )

        task_item: Dict[str, Any] = {
            "description": sentence[:500],
            "assignee_hint": _guess_assignee(sentence),
            "deadline_hint": _guess_deadline(sentence),
            "source": "rule_based",
        }
        if item.get("speaker_hint"):
            task_item["speaker_hint"] = item["speaker_hint"]
        if item.get("raw"):
            task_item["source_snippet"] = item["raw"][:180]

        if not _task_supported_by_transcript(task_item["description"], transcript):
            continue

        key = normalize_text(task_item["description"])[:140]
        if key in seen:
            continue
        seen.add(key)
        tasks.append(task_item)

    classifier_tasks = predict_trained_candidates(transcript, threshold=0.52, max_items=10)
    for cand in classifier_tasks:
        cand = dict(cand)
        cand["source"] = cand.get("source") or "trained_classifier"
        cand["description"] = _normalize_space(str(cand.get("description") or ""))
        cand["source_snippet"] = cand.get("source_snippet") or cand["description"]
        if not cand["description"] or _is_meta_task(cand["description"]):
            continue
        if not _task_supported_by_transcript(cand["description"], transcript):
            continue
        key = normalize_text(cand["description"])[:140]
        if key in seen:
            continue
        seen.add(key)
        tasks.append(cand)

    unique: List[Dict[str, Any]] = []
    seen_final: set[str] = set()
    for task in tasks:
        key = normalize_text(str(task.get("description") or ""))[:140]
        if not key or key in seen_final:
            continue
        seen_final.add(key)
        unique.append(task)

    return unique[:12]


# ---------------------------------------------------------------------------
# LLM parsing
# ---------------------------------------------------------------------------

def _collect_json_tasks(node: Any, tasks: List[Dict[str, Any]], transcript: str, candidate_map: dict[int, dict[str, Any]]) -> None:
    if isinstance(node, dict):
        candidate_id = node.get("candidate_id")
        desc = str(node.get("description") or node.get("task") or "").strip()
        speaker_hint = node.get("speaker_hint") or None
        source_snippet = node.get("source_snippet") or node.get("evidence") or None

        candidate = None
        if candidate_id is not None:
            try:
                candidate = candidate_map.get(int(candidate_id))
            except Exception:
                candidate = None

        if candidate is not None:
            desc = desc or str(candidate.get("description") or "")
            speaker_hint = speaker_hint or candidate.get("speaker_hint")
            source_snippet = source_snippet or candidate.get("source_snippet") or candidate.get("description")
        elif desc:
            best_match = None
            best_score = 0.0
            for cand in candidate_map.values():
                score = _candidate_similarity(desc, str(cand.get("description") or ""))
                if score > best_score:
                    best_match = cand
                    best_score = score
            if best_match is not None and best_score >= 0.58:
                candidate = best_match
                speaker_hint = speaker_hint or candidate.get("speaker_hint")
                source_snippet = source_snippet or candidate.get("source_snippet") or candidate.get("description")

        if candidate is not None or (desc and _looks_like_task(desc) and _task_supported_by_transcript(desc, transcript)):
            item: Dict[str, Any] = {
                "description": desc[:500],
                "assignee_hint": _normalize_assignee_hint(node.get("assignee_hint") or node.get("assignee") or (candidate or {}).get("assignee_hint")),
                "deadline_hint": _normalize_deadline_hint(node.get("deadline_hint") or node.get("deadline") or (candidate or {}).get("deadline_hint")),
                "speaker_hint": _normalize_space(str(speaker_hint)) if speaker_hint else None,
                "source": "openrouter",
                "candidate_id": int(candidate.get("id")) if candidate and candidate.get("id") is not None else (int(candidate_id) if candidate_id is not None and str(candidate_id).isdigit() else None),
            }
            if source_snippet:
                item["source_snippet"] = _normalize_space(str(source_snippet))[:180]
            tasks.append(item)

        for key, value in node.items():
            if key in {"description", "task", "assignee_hint", "assignee", "deadline_hint", "deadline", "speaker_hint", "source_snippet", "evidence", "candidate_id"}:
                continue
            _collect_json_tasks(value, tasks, transcript, candidate_map)

    elif isinstance(node, list):
        for item in node:
            _collect_json_tasks(item, tasks, transcript, candidate_map)

    elif isinstance(node, str):
        text = _normalize_space(node)
        if len(text) < 10 or _is_meta_task(text):
            return
        if _task_supported_by_transcript(text, transcript) and _looks_like_task(text):
            tasks.append(
                {
                    "description": text[:500],
                    "assignee_hint": _guess_assignee(text),
                    "deadline_hint": _guess_deadline(text),
                    "source": "openrouter_text",
                }
            )


def _parse_llm_output(
    raw: str,
    transcript: str,
    *,
    candidate_pool: list[dict[str, Any]],
    trace_id: str | None = None,
) -> tuple[List[Dict[str, Any]], dict[str, Any]]:
    candidate_map = _candidate_map(candidate_pool)
    debug: dict[str, Any] = {
        "raw": raw,
        "raw_preview": _preview(raw, 1200),
        "parse_stage": None,
        "parsed_tasks": 0,
    }

    candidates: List[str] = []

    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1].strip())

    start_obj = raw.find("{")
    end_obj = raw.rfind("}")
    if start_obj != -1 and end_obj > start_obj:
        candidates.append(raw[start_obj : end_obj + 1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        tasks: List[Dict[str, Any]] = []
        _collect_json_tasks(parsed, tasks, transcript, candidate_map)
        if tasks:
            unique: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for t in tasks:
                desc = normalize_text(str(t.get("description") or ""))[:160]
                if not desc or desc in seen:
                    continue
                if t.get("candidate_id") is None:
                    matched = None
                    best_score = 0.0
                    for cand in candidate_pool:
                        score = _candidate_similarity(str(t.get("description") or ""), str(cand.get("description") or ""))
                        if score > best_score:
                            matched = cand
                            best_score = score
                    if matched is None or best_score < 0.58:
                        continue
                seen.add(desc)
                unique.append(t)

            debug["parse_stage"] = "json_recursive"
            debug["parsed_tasks"] = len(unique)
            return unique, debug

    tasks: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = _normalize_space(line)
        if not line:
            continue

        low = normalize_text(line)
        if low in {"action items", "task", "tasks", "extracted action items"}:
            continue
        if low.startswith("[") and low.endswith("]"):
            continue
        if low.startswith("extracted action items"):
            continue

        if len(line) > 12 and _looks_like_task(line) and _task_supported_by_transcript(line, transcript):
            tasks.append(
                {
                    "description": line[:500],
                    "assignee_hint": _guess_assignee(line),
                    "deadline_hint": _guess_deadline(line),
                    "source": "openrouter_text",
                }
            )

    debug["parse_stage"] = "line_fallback"
    debug["parsed_tasks"] = len(tasks)
    return tasks, debug


# ---------------------------------------------------------------------------
# OpenRouter / NVIDIA calls
# ---------------------------------------------------------------------------

def _chat_completions_url(base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _backend_debug(provider: str, model: str, raw: str, parse_debug: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "raw_preview": _preview(raw, 1200),
        "parse_stage": parse_debug.get("parse_stage"),
        "parsed_tasks": parse_debug.get("parsed_tasks", 0),
        "fallback_used": False,
    }


def _candidate_payload(pool: list[dict[str, Any]]) -> str:
    simplified = [
        {
            "id": item.get("id"),
            "speaker_hint": item.get("speaker_hint"),
            "text": item.get("description"),
            "source": item.get("source"),
            "score": item.get("score"),
        }
        for item in pool
    ]
    return json.dumps(simplified, ensure_ascii=False, indent=2)


def _transcript_excerpt_for_prompt(transcript: str, limit: int = 5000) -> str:
    transcript = _normalize_space(transcript)
    return transcript[:limit] + ("..." if len(transcript) > limit else "")


def _provider_model_candidates(provider: str) -> list[str]:
    if provider == "openrouter":
        configured = _normalize_space(str(getattr(settings, "OPENROUTER_TASK_MODEL", "") or ""))
        candidates = [configured] if configured else []
        # Reliable router aliases first, then a small free-model fallback.
        for fallback in ["openrouter/auto", "openrouter/free", "meta-llama/llama-3.2-3b-instruct:free"]:
            if fallback not in candidates:
                candidates.append(fallback)
        return [c for c in candidates if c]

    configured = _normalize_space(str(getattr(settings, "NVIDIA_TASK_MODEL", "") or ""))
    candidates = [configured] if configured else []
    # Ordered by benchmark results on AMI data (avg_f1, reliability):
    #   google/gemma-3n-e4b-it    → f1=0.203, 13/13 ok  ← best reliable model
    #   qwen/qwen3-coder-480b-*   → f1=0.144, 13/13 ok
    #   google/gemma-3n-e2b-it    → f1=0.185, 13/13 ok  (slightly lower than e4b)
    for fallback in [
        "google/gemma-3n-e4b-it",
        "qwen/qwen3-coder-480b-a35b-instruct",
        "google/gemma-3n-e2b-it",
    ]:
        if fallback not in candidates:
            candidates.append(fallback)
    return [c for c in candidates if c]


async def _call_chat_backend(
    transcript: str,
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model_candidates: list[str],
    trace_id: str | None = None,
    meeting_ref: str | None = None,
    language: str = "en",
    duration_sec: float | None = None,
    transcript_confidence: float | None = None,
    candidate_pool: list[dict[str, Any]] | None = None,
    retries: int = 2,
) -> tuple[List[Dict[str, Any]], dict[str, Any]]:
    if not api_key:
        raise RuntimeError(f"{provider} API key is not set")

    candidate_pool = candidate_pool or []
    user_content = USER_TEMPLATE.format(
        meeting_ref=meeting_ref or "unknown",
        language=language or "en",
        duration_sec=duration_sec or 0.0,
        transcript_confidence=f"{transcript_confidence:.3f}" if transcript_confidence is not None else "unknown",
        candidate_json=_candidate_payload(candidate_pool),
        transcript_excerpt=_transcript_excerpt_for_prompt(transcript),
    )

    payload = {
        "temperature": 0,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/your-org/meeting-secretary"
        headers["X-Title"] = "Meeting Secretary"

    url = _chat_completions_url(base_url)
    last_error: Exception | None = None
    retryable_statuses = {408, 425, 429, 500, 502, 503, 504, 529}

    async with httpx.AsyncClient(timeout=60.0) as client:
        for model in model_candidates:
            payload["model"] = model
            for attempt in range(retries):
                try:
                    resp = await client.post(url, json=payload, headers=headers)

                    if resp.status_code == 404:
                        raise RuntimeError(f"{provider} model not found: {model}")

                    if resp.status_code == 429:
                        retry_after = resp.headers.get("Retry-After")
                        wait_sec = float(retry_after) if retry_after and retry_after.isdigit() else BASE_BACKOFF_SEC * (2**attempt)
                        wait_sec += random.uniform(0, 0.5)
                        logger.warning(
                            "[TASK][%s] %s 429 rate limit%s, retry in %.1fs (attempt %d/%d, model=%s)",
                            trace_id or "-",
                            provider.capitalize(),
                            f", Retry-After={retry_after}" if retry_after else "",
                            wait_sec,
                            attempt + 1,
                            retries,
                            model,
                        )
                        if attempt < retries - 1:
                            await asyncio.sleep(wait_sec)
                            continue
                        raise RuntimeError(f"{provider} rate limited (429)")

                    if resp.status_code in retryable_statuses:
                        wait_sec = BASE_BACKOFF_SEC * (2**attempt) + random.uniform(0, 0.5)
                        logger.warning(
                            "[TASK][%s] %s transient error %s, retry in %.1fs (attempt %d/%d, model=%s)",
                            trace_id or "-",
                            provider.capitalize(),
                            resp.status_code,
                            wait_sec,
                            attempt + 1,
                            retries,
                            model,
                        )
                        if attempt < retries - 1:
                            await asyncio.sleep(wait_sec)
                            continue
                        resp.raise_for_status()

                    if 400 <= resp.status_code < 500:
                        resp.raise_for_status()

                    data = resp.json()
                    actual_model = data.get("model", model)
                    raw = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or "[]"

                    logger.info("[TASK][%s] %s model used: %s", trace_id or "-", provider.capitalize(), actual_model)
                    logger.info("[TASK][%s] %s raw output:\n%s", trace_id or "-", provider.capitalize(), raw)

                    tasks, parse_debug = _parse_llm_output(raw, transcript, candidate_pool=candidate_pool, trace_id=trace_id)
                    for t in tasks:
                        t["model"] = actual_model
                        t["source"] = provider
                        if t.get("candidate_id") is not None:
                            try:
                                candidate = _candidate_map(candidate_pool).get(int(t["candidate_id"]))
                                if candidate is not None:
                                    t["speaker_hint"] = t.get("speaker_hint") or candidate.get("speaker_hint")
                                    t["source_snippet"] = t.get("source_snippet") or candidate.get("source_snippet")
                            except Exception:
                                pass

                    debug = _backend_debug(provider, actual_model, raw, parse_debug)
                    debug["used_model_candidates"] = model_candidates
                    return tasks, debug

                except Exception as exc:
                    last_error = exc
                    msg = str(exc).lower()
                    if "model not found" in msg or (isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404):
                        logger.warning("[TASK][%s] %s candidate model failed, trying next model: %s", trace_id or "-", provider, model)
                        break
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in retryable_statuses:
                        break
                    if attempt < retries - 1:
                        wait_sec = BASE_BACKOFF_SEC * (2**attempt) + random.uniform(0, 0.5)
                        logger.warning(
                            "[TASK][%s] %s error, retry in %.1fs (attempt %d/%d, model=%s): %s",
                            trace_id or "-",
                            provider.capitalize(),
                            wait_sec,
                            attempt + 1,
                            retries,
                            model,
                            exc,
                        )
                        await asyncio.sleep(wait_sec)
                        continue
                    break

    raise RuntimeError(f"{provider} failed after retries: {last_error}")


async def _call_openrouter(
    transcript: str,
    *,
    trace_id: str | None = None,
    meeting_ref: str | None = None,
    language: str = "en",
    duration_sec: float | None = None,
    transcript_confidence: float | None = None,
    candidate_pool: list[dict[str, Any]] | None = None,
) -> tuple[List[Dict[str, Any]], dict[str, Any]]:
    api_key: str = getattr(settings, "OPENROUTER_API_KEY", "") or ""
    return await _call_chat_backend(
        transcript,
        provider="openrouter",
        base_url=_OPENROUTER_BASE_URL,
        api_key=api_key,
        model_candidates=_provider_model_candidates("openrouter"),
        trace_id=trace_id,
        meeting_ref=meeting_ref,
        language=language,
        duration_sec=duration_sec,
        transcript_confidence=transcript_confidence,
        candidate_pool=candidate_pool,
        retries=MAX_OPENROUTER_RETRIES,
    )


async def _call_nvidia(
    transcript: str,
    *,
    trace_id: str | None = None,
    meeting_ref: str | None = None,
    language: str = "en",
    duration_sec: float | None = None,
    transcript_confidence: float | None = None,
    candidate_pool: list[dict[str, Any]] | None = None,
) -> tuple[List[Dict[str, Any]], dict[str, Any]]:
    api_key: str = getattr(settings, "NVIDIA_API_KEY", "") or ""
    base_url: str = getattr(settings, "NVIDIA_BASE_URL", _NVIDIA_BASE_URL_DEFAULT) or _NVIDIA_BASE_URL_DEFAULT
    return await _call_chat_backend(
        transcript,
        provider="nvidia",
        base_url=base_url,
        api_key=api_key,
        model_candidates=_provider_model_candidates("nvidia"),
        trace_id=trace_id,
        meeting_ref=meeting_ref,
        language=language,
        duration_sec=duration_sec,
        transcript_confidence=transcript_confidence,
        candidate_pool=candidate_pool,
        retries=MAX_NVIDIA_RETRIES,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_tasks(
    transcript: str,
    *,
    return_debug: bool = False,
    trace_id: str | None = None,
    meeting_ref: str | None = None,
    language: str = "en",
    duration_sec: float | None = None,
    transcript_confidence: float | None = None,
) -> List[Dict[str, Any]] | tuple[List[Dict[str, Any]], dict[str, Any]]:
    transcript = _normalize_space(transcript)
    requested_provider: str = getattr(settings, "TASK_PROVIDER", "openrouter") or "openrouter"
    fallback_tasks = _extract_tasks_simple(transcript)
    candidates = _candidate_pool(transcript)

    debug: dict[str, Any] = {
        "requested_provider": requested_provider,
        "provider": requested_provider,
        "trace_id": trace_id,
        "fallback_tasks": len(fallback_tasks),
        "candidate_pool": len(candidates),
        "model": None,
        "raw_preview": None,
        "parse_stage": None,
        "fallback_used": requested_provider == "rules",
        "fallback_merged": False,
        "conservative_mode": False,
    }

    if not transcript:
        debug["reason"] = "empty transcript"
        return ([], debug) if return_debug else []

    words_count = len(transcript.split())
    short_or_noisy = (
        (duration_sec is not None and duration_sec < 180)
        or words_count < 40
        or (transcript_confidence is not None and transcript_confidence < 0.60)
    )

    if short_or_noisy and requested_provider != "rules":
        logger.info(
            "[TASK][%s] Conservative mode enabled: short/noisy transcript, using rule-based fallback only",
            trace_id or "-",
        )
        debug["conservative_mode"] = True
        debug["fallback_used"] = True
        debug["provider"] = "rules"
        debug["model"] = "rules"
        result = fallback_tasks or candidates
        result = result[:12]
        return (result, debug) if return_debug else result

    if requested_provider == "rules":
        logger.info("[TASK][%s] Using rule-based extraction", trace_id or "-")
        debug["provider"] = "rules"
        debug["model"] = "rules"
        result = fallback_tasks or candidates
        result = result[:12]
        return (result, debug) if return_debug else result

    provider_order: list[str] = []
    if requested_provider in {"openrouter", "nvidia"}:
        provider_order.append(requested_provider)
    else:
        provider_order.append("openrouter")

    if requested_provider != "nvidia" and getattr(settings, "NVIDIA_API_KEY", ""):
        provider_order.append("nvidia")
    if requested_provider != "openrouter" and getattr(settings, "OPENROUTER_API_KEY", ""):
        provider_order.append("openrouter")
    if not provider_order:
        provider_order = ["openrouter", "nvidia"]

    seen_providers: set[str] = set()
    llm_tasks: List[Dict[str, Any]] = []
    llm_debug: dict[str, Any] = {}
    last_error: Exception | None = None

    for provider_name in provider_order:
        if provider_name in seen_providers:
            continue
        seen_providers.add(provider_name)

        try:
            if provider_name == "openrouter":
                llm_tasks, llm_debug = await _call_openrouter(
                    transcript,
                    trace_id=trace_id,
                    meeting_ref=meeting_ref,
                    language=language,
                    duration_sec=duration_sec,
                    transcript_confidence=transcript_confidence,
                    candidate_pool=candidates,
                )
            elif provider_name == "nvidia":
                llm_tasks, llm_debug = await _call_nvidia(
                    transcript,
                    trace_id=trace_id,
                    meeting_ref=meeting_ref,
                    language=language,
                    duration_sec=duration_sec,
                    transcript_confidence=transcript_confidence,
                    candidate_pool=candidates,
                )
            else:
                continue

            debug.update(llm_debug)
            debug["provider"] = llm_debug.get("provider", provider_name)
            debug["model"] = llm_debug.get("model")
            last_error = None

            if llm_tasks:
                break

            logger.info("[TASK][%s] %s returned no tasks; trying next fallback", trace_id or "-", provider_name)
        except Exception as exc:
            last_error = exc
            debug["error"] = str(exc)
            debug["provider_error"] = provider_name
            logger.warning("[TASK][%s] %s failed: %s", trace_id or "-", provider_name, exc)
            continue

    if not llm_tasks:
        if last_error is not None:
            debug["error"] = str(last_error)
        debug["fallback_used"] = True
        logger.info("[TASK][%s] Falling back to rule-based extraction", trace_id or "-")
        debug["provider"] = "rules"
        debug["model"] = "rules"
        result = fallback_tasks or candidates
        result = result[:12]
        return (result, debug) if return_debug else result

    debug["llm_tasks"] = len(llm_tasks)

    combined = llm_tasks if len(llm_tasks) >= 2 else llm_tasks + fallback_tasks
    debug["fallback_merged"] = len(llm_tasks) < 2 and len(fallback_tasks) > 0
    debug["fallback_used"] = debug["fallback_used"] or debug["fallback_merged"] or debug["provider"] != requested_provider

    unique: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for t in combined:
        key = _normalize_space((t.get("description") or "").lower())[:140]
        if key and key not in seen:
            seen.add(key)
            unique.append(t)

    result = unique[:20]
    debug["final_tasks"] = len(result)
    debug["provider"] = debug.get("provider") or requested_provider

    return (result, debug) if return_debug else result


def extract_tasks_rule_based(transcript: str) -> List[Dict[str, Any]]:
    return _extract_tasks_simple(transcript)
