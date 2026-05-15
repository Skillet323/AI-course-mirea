"""Task extraction with OpenRouter + NVIDIA fallback + strict rule-based fallback.

Returned task dicts are normalized to:
    {
        "description": str,
        "assignee_hint": str | None,
        "deadline_hint": str | None,
        "speaker_hint": str | None,      # optional
        "source": "openrouter" | "nvidia" | "openrouter_text" | "rule_based" | "trained_classifier",
        "source_snippet": str | None,    # optional
        "model": str | None,             # only for LLM outputs
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings
from .task_classifier import predict_candidates as predict_trained_candidates

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_NVIDIA_BASE_URL_DEFAULT = "https://integrate.api.nvidia.com/v1"
MAX_OPENROUTER_RETRIES = 3
MAX_NVIDIA_RETRIES = 3
BASE_BACKOFF_SEC = 2.5

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "had", "are", "was", "were",
    "will", "would", "could", "should", "need", "needs", "please", "about", "into", "onto", "then",
    "than", "them", "they", "their", "there", "here", "what", "when", "where", "why", "how", "who",
    "whom", "which", "your", "our", "you", "we", "uh", "um", "mm", "yeah", "okay", "ok", "right",
    "just", "also", "still", "very", "really", "maybe"
}


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
    """Split speaker-labeled lines such as `SPEAKER_00: hello` or `Alice: hello`."""
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


def _task_supported_by_transcript(description: str, transcript: str, min_overlap: float = 0.10) -> bool:
    """Keep only tasks with some lexical support in the current transcript."""
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
        r"(?:for|to|assigned to|by)\s+([A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)?)",
        r"(?:для|от|к)\s+([A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)?)",
        r"@([\w.-]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, sentence, flags=re.IGNORECASE)
        if not m:
            continue
        candidate = _normalize_space(m.group(1))
        low = normalize_text(candidate)
        if not candidate or low in _ASSIGNEE_DEADLINE_TOKENS:
            continue
        return candidate
    return None


def _looks_like_task(sentence: str) -> bool:
    """Conservative rule-based detection for fallback."""
    s = normalize_text(sentence)

    deny = [
        "agenda",
        "project manager",
        "we re developing",
        "we are developing",
        "first meeting",
        "icebreaker",
        "favourite animal",
        "favourite characteristic",
        "white board",
        "design stages",
        "finance",
        "marketing",
        "introduction",
        "introduce ourselves",
        "introduce self",
        "meeting agenda",
        "good morning",
        "hello everybody",
        "i am",
        "my name is",
    ]
    if any(d in s for d in deny):
        return False

    markers = [
        # English action verbs / modal phrases
        "should",
        "must",
        "need to",
        "needs to",
        "please",
        "action item",
        "action point",
        "task",
        "to do",
        "todo",
        "follow up",
        "follow-up",
        "let s",
        "let us",
        "have to",
        "required to",
        "going to",
        "will have",
        "will need",
        "will be responsible",
        "take care of",
        "make sure",
        "ensure that",
        "prepare",
        "design",
        "develop",
        "implement",
        "create",
        "write",
        "send",
        "schedule",
        "arrange",
        "contact",
        "coordinate",
        "review",
        "check",
        "update",
        "look into",
        "look at",
        "work on",
        "put together",
        "come up with",
        "figure out",
        "set up",
        "keep track",
        "make a decision",
        "responsible for",
        "in charge of",
        "assigned to",
        "by next",
        "before next",
        # Russian
        "нужно",
        "надо",
        "нужна",
        "нужен",
        "сделать",
        "подготовить",
        "проверить",
        "отправить",
        "согласовать",
        "обновить",
        "выполнить",
        "завершить",
        "доработать",
        "назначить",
        "созвониться",
        "позвонить",
        "разработать",
        "написать",
        "составить",
        "организовать",
        "предоставить",
        "убедиться",
    ]
    return any(marker in s for marker in markers)


def _is_meta_task(text: str) -> bool:
    """Filter out obvious model meta-output or meeting narration."""
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
    """Keep only plausible specific names/roles and drop generic pseudo-assignees."""
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
        out["source_snippet"] = source_snippet[:120]
    if item.get("model"):
        out["model"] = item["model"]
    if item.get("score") is not None:
        out["score"] = item.get("score")

    tasks.append(out)


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


def _extract_tasks_simple(transcript: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", transcript)

    for raw in sentences:
        speaker, body = _split_speaker_prefix(raw)
        sentence = _normalize_space(body)
        if not sentence or len(sentence) < 12:
            continue
        if not _looks_like_task(sentence):
            continue

        sentence = re.sub(
            r"^(?:ну|okay|ok|please|let's|we should|we need to|нужно|надо)\s*,?\s*",
            "",
            sentence,
            flags=re.IGNORECASE,
        )

        item: Dict[str, Any] = {
            "description": sentence[:500],
            "assignee_hint": _guess_assignee(sentence),
            "deadline_hint": _guess_deadline(sentence),
            "source": "rule_based",
        }
        if speaker:
            item["speaker_hint"] = speaker

        if _task_supported_by_transcript(item["description"], transcript):
            tasks.append(item)

    classifier_tasks = predict_trained_candidates(transcript, threshold=0.70, max_items=8)
    for cand in classifier_tasks:
        cand["source"] = "trained_classifier"
        tasks.append(cand)

    unique: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for t in tasks:
        key = _normalize_space(t["description"].lower())[:120]
        if key and key not in seen:
            seen.add(key)
            unique.append(t)

    return unique


# ---------------------------------------------------------------------------
# LLM parsing helpers
# ---------------------------------------------------------------------------


def _collect_json_tasks(node: Any, tasks: List[Dict[str, Any]], transcript: str, source: str) -> None:
    """Recursively collect task-like objects from arbitrarily nested JSON."""
    if isinstance(node, dict):
        desc = str(node.get("description") or node.get("task") or "").strip()
        if desc and not _is_meta_task(desc) and _task_supported_by_transcript(desc, transcript):
            item: Dict[str, Any] = {
                "description": desc[:500],
                "assignee_hint": _normalize_assignee_hint(node.get("assignee_hint") or node.get("assignee")),
                "deadline_hint": _normalize_deadline_hint(node.get("deadline_hint") or node.get("deadline")),
                "source": source,
            }
            if node.get("speaker_hint"):
                item["speaker_hint"] = _normalize_space(str(node["speaker_hint"]))
            if node.get("source_snippet"):
                item["source_snippet"] = str(node["source_snippet"])[:120]
            tasks.append(item)

        for key, value in node.items():
            if key in {"description", "task", "assignee_hint", "assignee", "deadline_hint", "deadline", "source_snippet", "evidence", "speaker_hint"}:
                continue
            _collect_json_tasks(value, tasks, transcript, source)

    elif isinstance(node, list):
        for item in node:
            _collect_json_tasks(item, tasks, transcript, source)

    elif isinstance(node, str):
        text = _normalize_space(node)
        if len(text) < 10 or _is_meta_task(text):
            return
        if _task_supported_by_transcript(text, transcript):
            tasks.append(
                {
                    "description": text[:500],
                    "assignee_hint": _guess_assignee(text),
                    "deadline_hint": _guess_deadline(text),
                    "source": source,
                }
            )


def _parse_llm_output(raw: str, transcript: str) -> tuple[List[Dict[str, Any]], dict[str, Any]]:
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

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        tasks: List[Dict[str, Any]] = []
        _collect_json_tasks(parsed, tasks, transcript, source="openrouter")
        if tasks:
            unique: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for t in tasks:
                desc = _normalize_space((t.get("description") or "").lower())[:120]
                if not desc or desc in seen:
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

        if len(line) > 15 and _looks_like_task(line) and _task_supported_by_transcript(line, transcript):
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

_SYSTEM_PROMPT = (
    "You are a strict action-item extractor for meeting transcripts.\n"
    "Return ONLY a valid JSON array. No prose. No markdown. No explanation.\n"
    "If no valid tasks exist, return [].\n\n"
    "Rules:\n"
    "- Extract only tasks that are clearly intended to happen AFTER the meeting.\n"
    "- Do NOT extract introductions, agenda items, discussion topics, or summaries.\n"
    "- Do NOT invent people, names, roles, deadlines, or tasks.\n"
    "- If the transcript is short, noisy, or mostly off-topic, prefer returning [].\n"
    "- Keep descriptions concrete and short.\n"
    "- Use assignee_hint only if the person/role is actually mentioned or strongly implied in the transcript.\n"
    "- Use deadline_hint only if explicitly stated or very clearly implied.\n"
    "- speaker_hint should be copied only from actual speaker labels found in the transcript (e.g. SPEAKER_00, A). Do not turn them into human names.\n\n"
    "Output schema for each item:\n"
    "{"
    '"description": "string", '
    '"assignee_hint": "string or null", '
    '"deadline_hint": "string or null", '
    '"speaker_hint": "string or null", '
    '"source_snippet": "string or null"'
    "}"
)

_USER_TEMPLATE = (
    "Extract action items from the meeting transcript below.\n\n"
    "Meeting ID: {meeting_ref}\n"
    "Language: {language}\n"
    "Duration: {duration_sec:.1f} seconds\n"
    "Transcript confidence: {transcript_confidence}\n\n"
    "Transcript:\n"
    "{transcript}\n\n"
    "Return only a JSON array."
)


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


async def _call_chat_backend(
    transcript: str,
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    trace_id: str | None = None,
    meeting_ref: str | None = None,
    language: str = "en",
    duration_sec: float | None = None,
    transcript_confidence: float | None = None,
    retries: int = 3,
) -> tuple[List[Dict[str, Any]], dict[str, Any]]:
    if not api_key:
        raise RuntimeError(f"{provider} API key is not set")

    user_content = _USER_TEMPLATE.format(
        meeting_ref=meeting_ref or "unknown",
        language=language or "en",
        duration_sec=duration_sec or 0.0,
        transcript_confidence=f"{transcript_confidence:.3f}" if transcript_confidence is not None else "unknown",
        transcript=transcript[:8000],
    )

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
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
        for attempt in range(retries):
            try:
                resp = await client.post(url, json=payload, headers=headers)

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_sec = float(retry_after)
                        except ValueError:
                            wait_sec = BASE_BACKOFF_SEC * (2**attempt)
                    else:
                        wait_sec = BASE_BACKOFF_SEC * (2**attempt)
                    wait_sec += random.uniform(0, 0.75)
                    logger.warning(
                        "[TASK][%s] %s 429 rate limit%s, retry in %.1fs (attempt %d/%d)",
                        trace_id or "-",
                        provider.capitalize(),
                        f", Retry-After={retry_after}" if retry_after else "",
                        wait_sec,
                        attempt + 1,
                        retries,
                    )
                    if attempt < retries - 1:
                        await asyncio.sleep(wait_sec)
                        continue
                    raise RuntimeError(f"{provider} rate limited (429)")

                if resp.status_code in retryable_statuses:
                    wait_sec = BASE_BACKOFF_SEC * (2**attempt) + random.uniform(0, 0.75)
                    logger.warning(
                        "[TASK][%s] %s transient error %s, retry in %.1fs (attempt %d/%d)",
                        trace_id or "-",
                        provider.capitalize(),
                        resp.status_code,
                        wait_sec,
                        attempt + 1,
                        retries,
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

                tasks, parse_debug = _parse_llm_output(raw, transcript)
                for t in tasks:
                    t["model"] = actual_model
                    t["source"] = provider

                debug = _backend_debug(provider, actual_model, raw, parse_debug)
                return tasks, debug

            except Exception as exc:
                last_error = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in retryable_statuses:
                    break
                if attempt < retries - 1:
                    wait_sec = BASE_BACKOFF_SEC * (2**attempt) + random.uniform(0, 0.75)
                    logger.warning(
                        "[TASK][%s] %s error, retry in %.1fs (attempt %d/%d): %s",
                        trace_id or "-",
                        provider.capitalize(),
                        wait_sec,
                        attempt + 1,
                        retries,
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
) -> tuple[List[Dict[str, Any]], dict[str, Any]]:
    api_key: str = getattr(settings, "OPENROUTER_API_KEY", "") or ""
    model: str = getattr(settings, "OPENROUTER_TASK_MODEL", "openrouter/free") or "openrouter/free"
    return await _call_chat_backend(
        transcript,
        provider="openrouter",
        base_url=_OPENROUTER_BASE_URL,
        api_key=api_key,
        model=model,
        trace_id=trace_id,
        meeting_ref=meeting_ref,
        language=language,
        duration_sec=duration_sec,
        transcript_confidence=transcript_confidence,
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
) -> tuple[List[Dict[str, Any]], dict[str, Any]]:
    api_key: str = getattr(settings, "NVIDIA_API_KEY", "") or ""
    model: str = getattr(settings, "NVIDIA_TASK_MODEL", "meta/llama-4-maverick-17b-128e-instruct") or "meta/llama-4-maverick-17b-128e-instruct"
    base_url: str = getattr(settings, "NVIDIA_BASE_URL", _NVIDIA_BASE_URL_DEFAULT) or _NVIDIA_BASE_URL_DEFAULT
    return await _call_chat_backend(
        transcript,
        provider="nvidia",
        base_url=base_url,
        api_key=api_key,
        model=model,
        trace_id=trace_id,
        meeting_ref=meeting_ref,
        language=language,
        duration_sec=duration_sec,
        transcript_confidence=transcript_confidence,
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

    debug: dict[str, Any] = {
        "requested_provider": requested_provider,
        "provider": requested_provider,
        "trace_id": trace_id,
        "fallback_tasks": len(fallback_tasks),
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
        return (fallback_tasks, debug) if return_debug else fallback_tasks

    if requested_provider == "rules":
        logger.info("[TASK][%s] Using rule-based extraction", trace_id or "-")
        debug["provider"] = "rules"
        debug["model"] = "rules"
        return (fallback_tasks, debug) if return_debug else fallback_tasks

    provider_order: list[str] = []
    normalized_requested = requested_provider.lower().strip()
    if normalized_requested in {"openrouter", "nvidia"}:
        provider_order.append(normalized_requested)
    else:
        provider_order.append("openrouter")

    if normalized_requested == "openrouter":
        if getattr(settings, "NVIDIA_API_KEY", ""):
            provider_order.append("nvidia")
    elif normalized_requested == "nvidia":
        if getattr(settings, "OPENROUTER_API_KEY", ""):
            provider_order.append("openrouter")
    else:
        if getattr(settings, "OPENROUTER_API_KEY", "") and "openrouter" not in provider_order:
            provider_order.append("openrouter")
        if getattr(settings, "NVIDIA_API_KEY", "") and "nvidia" not in provider_order:
            provider_order.append("nvidia")

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
                )
            elif provider_name == "nvidia":
                llm_tasks, llm_debug = await _call_nvidia(
                    transcript,
                    trace_id=trace_id,
                    meeting_ref=meeting_ref,
                    language=language,
                    duration_sec=duration_sec,
                    transcript_confidence=transcript_confidence,
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
        return (fallback_tasks, debug) if return_debug else fallback_tasks

    debug["llm_tasks"] = len(llm_tasks)

    combined = llm_tasks if len(llm_tasks) >= 2 else llm_tasks + fallback_tasks
    debug["fallback_merged"] = len(llm_tasks) < 2 and len(fallback_tasks) > 0
    debug["fallback_used"] = debug["fallback_used"] or debug["fallback_merged"] or debug["provider"] != requested_provider

    unique: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for t in combined:
        key = _normalize_space((t.get("description") or "").lower())[:120]
        if key and key not in seen:
            seen.add(key)
            unique.append(t)

    result = unique[:20]
    debug["final_tasks"] = len(result)
    debug["provider"] = debug.get("provider") or requested_provider

    return (result, debug) if return_debug else result


def extract_tasks_rule_based(transcript: str) -> List[Dict[str, Any]]:
    return _extract_tasks_simple(transcript)
