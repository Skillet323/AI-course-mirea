from __future__ import annotations
import argparse
import csv
import dataclasses
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
import httpx

# =============================================================================
# Configuration
# =============================================================================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MAX_TOKENS = 900
DEFAULT_TIMEOUT = 240.0  # PATCH 7: Increased for slower models / retries
SYSTEM_PROMPT = (
 "You are a strict action-item extractor for meeting transcripts.\n "
 "Return ONLY a valid JSON array. No prose. No markdown. No explanation.\n "
 "If no valid tasks exist, return [].\n\n "
 "Rules:\n "
 "- Extract only tasks that are clearly intended to happen AFTER the meeting.\n "
 "- Do NOT extract introductions, agenda items, discussion topics, or summaries.\n "
 "- Do NOT invent people, names, roles, deadlines, or tasks.\n "
 "- If the transcript is short, noisy, or mostly off-topic, prefer returning [].\n "
 "- Keep descriptions concrete and short.\n "
 "- assignee_hint must be a real name/role actually mentioned or strongly implied in the transcript.\n "
 "- deadline_hint must be explicit or very clearly implied.\n "
 "- speaker_hint, if present, must copy an actual speaker label from the transcript (e.g. SPEAKER_00, A). Do not turn it into a human name.\n\n "
 "Output schema for each item:\n "
'{'
' "description ":  "string ", '
' "assignee_hint ":  "string or null ", '
' "deadline_hint ":  "string or null ", '
' "speaker_hint ":  "string or null ", '
' "source_snippet ":  "string or null "'
'}'
)
USER_TEMPLATE = (
"Extract action items from the meeting transcript below.\n\n"
"Meeting ID: {meeting_ref}\n"
"Language: {language}\n"
"Duration: {duration_sec:.1f} seconds\n\n"
"Transcript:\n"
"{transcript}\n\n"
"Return only a JSON array."
)
STOPWORDS = {
 "a ",  "an ",  "and ",  "are ",  "as ",  "at ",  "be ",  "by ",  "for ",  "from ",  "have ",  "has ",  "had ",
 "he ",  "her ",  "him ",  "his ",  "i ",  "if ",  "in ",  "into ",  "is ",  "it ",  "its ",  "me ",  "my ",
 "of ",  "on ",  "or ",  "our ",  "she ",  "so ",  "that ",  "the ",  "their ",  "them ",  "then ",  "there ",
 "these ",  "they ",  "this ",  "to ",  "we ",  "were ",  "what ",  "when ",  "where ",  "which ",  "who ",
 "will ",  "with ",  "would ",  "you ",  "your ",  "um ",  "uh ",  "okay ",  "ok ",  "right ",  "yeah ",
 "just ",  "also ",  "still ",  "very ",  "really ",  "maybe ",  "please ",  "could ",  "should ",  "need ",
 "needs ",  "want ",  "wanna ",
}

# =============================================================================
# PATCH 1: add after STOPWORDS
# =============================================================================
NULL_LIKE = {"", "null", "none", "n/a", "na", "nil", "undefined"}

def clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = normalize_space(str(value))
    if normalize_text(text) in NULL_LIKE:
        return None
    return text or None

# =============================================================================
# Text Processing Helpers
# =============================================================================
def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    return [t for t in text.split() if t and t not in STOPWORDS and len(t) > 2]

def is_meta_task(text: str) -> bool:
    s = normalize_text(text)
    bad_phrases = [
        "review and summarize action items ", "extract action items from the meeting transcript ",
        "meeting transcript ", "summarize action items ", "introduce participants ",
        "confirm meeting agenda ", "project goal and objectives ", "outline the project structure ",
        "describe the functional design process ", "explain the tool training exercise ",
        "introduce the user interface design approach ", "confirm attendance ",
        "review current remote control features ", "start the meeting ", "confirm everyone is ready ",
        "read the entire transcript first ", "meeting id ", "language ", "duration ",
    ]
    return any(p in s for p in bad_phrases)

def parse_gold_dir(gold_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fp in sorted(gold_dir.glob("*.json")):
        with fp.open("r", encoding="utf-8") as f:
            item = json.load(f)
        if not isinstance(item, dict):
            continue
        item["_source_file"] = fp.name
        records.append(item)
    return records

def _candidate_texts(raw: str) -> list[str]:
    candidates: list[str] = []
    raw = raw or ""
    for m in re.finditer(r"`(?:json)?\s*(.*?)\s*`", raw, flags=re.DOTALL | re.IGNORECASE):
        block = m.group(1).strip()
        if block:
            candidates.append(block)

    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1].strip())

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1].strip())

    candidates.append(raw.strip())
    return candidates

# =============================================================================
# PATCH 2: replace entire _collect_tasks function
# =============================================================================
def _collect_tasks(node: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        desc = clean_optional_text(node.get("description") or node.get("task"))

        if desc and not is_meta_task(desc):
            out.append(
                {
                    "description": desc[:500],
                    "assignee_hint": clean_optional_text(
                        node.get("assignee_hint") or node.get("assignee")
                    ),
                    "deadline_hint": clean_optional_text(
                        node.get("deadline_hint") or node.get("deadline")
                    ),
                    "speaker_hint": clean_optional_text(
                        node.get("speaker_hint") or node.get("speaker")
                    ),
                    "source_snippet": clean_optional_text(
                        node.get("source_snippet") or node.get("evidence")
                    ),
                }
            )

        for key, value in node.items():
            if key in {
                "description", "task", "assignee_hint", "assignee",
                "deadline_hint", "deadline", "speaker_hint", "speaker",
                "source_snippet", "evidence",
            }:
                continue
            _collect_tasks(value, out)

    elif isinstance(node, list):
        for item in node:
            _collect_tasks(item, out)

    elif isinstance(node, str):
        text = clean_optional_text(node)
        if text and len(text) >= 10 and not is_meta_task(text):
            out.append(
                {
                    "description": text[:500],
                    "assignee_hint": None,
                    "deadline_hint": None,
                    "speaker_hint": None,
                    "source_snippet": None,
                }
            )

# =============================================================================
# PATCH 3: replace entire parse_tasks_from_raw function
# =============================================================================
def parse_tasks_from_raw(raw: str) -> tuple[list[dict[str, Any]], str]:
    for candidate in _candidate_texts(raw):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        tasks: list[dict[str, Any]] = []
        _collect_tasks(parsed, tasks)

        unique: list[dict[str, Any]] = []
        seen: set[str] = set()

        for t in tasks:
            desc = normalize_space(t.get("description", ""))
            key = normalize_text(desc)[:120]
            if not desc or key in seen:
                continue
            seen.add(key)
            unique.append(t)

        # IMPORTANT: even [] is a successful JSON parse
        return unique, "json"

    # -------------------------------------------------------------------------
    # fallback line parser
    # -------------------------------------------------------------------------
    tasks = []
    for line in raw.splitlines():
        line = normalize_space(line)
        if not line or len(line) < 10:
            continue
        low = normalize_text(line)
        if low in {"action items", "task", "tasks", "extracted action items"}:
            continue
        if low.startswith("extracted action items"):
            continue
        if line[0].isdigit():
            line = re.sub(r"^\d+[\).\-\s]+", "", line).strip()
        if len(line) >= 10 and not is_meta_task(line):
            tasks.append(
                {
                    "description": line[:500],
                    "assignee_hint": None,
                    "deadline_hint": None,
                    "speaker_hint": None,
                    "source_snippet": None,
                }
            )

    unique = []
    seen = set()
    for t in tasks:
        key = normalize_text(t["description"])[:120]
        if key and key not in seen:
            seen.add(key)
            unique.append(t)

    if unique:
        return unique, "line"

    return [], "empty"

def _sim(a: str, b: str) -> float:
    a_n, b_n = normalize_text(a), normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    seq = SequenceMatcher(None, a_n, b_n).ratio()
    a_tokens, b_tokens = set(tokenize(a_n)), set(tokenize(b_n))
    if not a_tokens or not b_tokens:
        return seq
    jacc = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    return 0.55 * jacc + 0.45 * seq

def evaluate_tasks(
    pred_tasks: list[dict[str, Any]],
    gold_tasks: list[dict[str, Any]],
    threshold: float = 0.28,
) -> dict[str, Any]:
    pred_tasks = pred_tasks or []
    gold_tasks = gold_tasks or []
    matches: dict[int, int] = {}
    used_gold: set[int] = set()
    gold_descs = [normalize_space(g.get("description") or g.get("task") or "") for g in gold_tasks]

    for i, pred in enumerate(pred_tasks):
        pdesc = normalize_space(pred.get("description") or pred.get("task") or "")
        best_idx, best_score = -1, 0.0
        for j, gdesc in enumerate(gold_descs):
            if j in used_gold:
                continue
            score = _sim(pdesc, gdesc)
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

    assignee_correct = 0
    deadline_correct = 0
    assignment_checks = 0
    deadline_checks = 0

    for pred_idx, gold_idx in matches.items():
        pred = pred_tasks[pred_idx]
        gold = gold_tasks[gold_idx]

        pred_assignee = normalize_text(pred.get("assignee_hint") or pred.get("assignee") or "")
        gold_assignee = normalize_text(gold.get("assignee_hint") or gold.get("assignee") or "")
        pred_deadline = normalize_text(pred.get("deadline_hint") or pred.get("deadline") or "")
        gold_deadline = normalize_text(gold.get("deadline_hint") or gold.get("deadline") or "")

        if pred_assignee or gold_assignee:
            assignment_checks += 1
            if not pred_assignee and not gold_assignee:
                assignee_correct += 1
            elif pred_assignee and gold_assignee and (
                pred_assignee == gold_assignee
                or pred_assignee in gold_assignee
                or gold_assignee in pred_assignee
            ):
                assignee_correct += 1

        if pred_deadline or gold_deadline:
            deadline_checks += 1
            if not pred_deadline and not gold_deadline:
                deadline_correct += 1
            elif pred_deadline and gold_deadline and (
                pred_deadline == gold_deadline
                or pred_deadline in gold_deadline
                or gold_deadline in pred_deadline
            ):
                deadline_correct += 1

    assignee_accuracy = assignee_correct / assignment_checks if assignment_checks else None
    deadline_accuracy = deadline_correct / deadline_checks if deadline_checks else None
    hallucination_rate = sum(
        1 for t in pred_tasks if len(normalize_space(t.get("description", "")).split()) < 3
    ) / max(1, len(pred_tasks))

    return {
        "task_set_f1": f1,
        "task_set_precision": precision,
        "task_set_recall": recall,
        "assignee_accuracy": assignee_accuracy,
        "deadline_accuracy": deadline_accuracy,
        "hallucination_rate": hallucination_rate,
        "predicted_tasks": len(pred_tasks),
        "gold_tasks": len(gold_tasks),
        "matched_tasks": tp,
        "assignment_checks": assignment_checks,
        "deadline_checks": deadline_checks,
    }

# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class RequestResult:
    status: str
    http_status: int | None = None
    error: str | None = None
    retry_after: float | None = None
    response_model: str | None = None
    raw: str | None = None
    raw_preview: str | None = None
    parse_stage: str | None = None
    tasks: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    latency_sec: float | None = None

# =============================================================================
# API Interaction
# =============================================================================
def _get_headers(provider: str, api_key: str) -> dict[str, str]:
    if provider == "nvidia":
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://example.com"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Meeting Secretary Benchmark"),
    }

def _get_url(provider: str) -> str:
    return NVIDIA_URL if provider == "nvidia" else OPENROUTER_URL

def _post_request(client: httpx.Client, provider: str, api_key: str, payload: dict[str, Any]) -> httpx.Response:
    return client.post(_get_url(provider), json=payload, headers=_get_headers(provider, api_key))

def _parse_retry_after(value: str | None) -> float | None:
    """Parse Retry-After header: supports both delay-seconds and HTTP-date."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %y %H:%M:%S %Z"):
        try:
            parsed = time.strptime(value.strip(), fmt)
            return max(0.0, time.mktime(parsed) - time.time())
        except ValueError:
            continue
    return None

def probe_model(
    client: httpx.Client,
    api_key: str,
    model: str,
    provider: str = "openrouter",
    verbose: bool = False,
) -> RequestResult:
    payload = {
        "model": model, "temperature": 0, "max_tokens": 1,
        "messages": [
            {"role": "system", "content": "Reply with OK."},
            {"role": "user", "content": "OK"},
        ],
    }
    try:
        resp = _post_request(client, provider, api_key, payload)
    except Exception as exc:
        return RequestResult(status="error", error=str(exc))

    if resp.status_code == 429:
        retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
        if verbose:
            print(f"  [RATE LIMIT] {model}: retry_after={retry_after}s")
        return RequestResult(status="rate_limited", http_status=429, error="rate_limited", retry_after=retry_after)

    if resp.status_code == 403:
        return RequestResult(status="forbidden", http_status=403, error="forbidden")

    if resp.status_code != 200:
        err_preview = resp.text[:200].replace("\n", "  ")
        if verbose:
            print(f"  [ERROR {resp.status_code}] {model}: {err_preview}")
        return RequestResult(status="error", http_status=resp.status_code, error=err_preview)

    try:
        data = resp.json()
    except Exception as exc:
        return RequestResult(status="error", http_status=200, error=f"bad_json:{exc}")

    return RequestResult(status="ok", http_status=200, response_model=data.get("model") or model)

def request_tasks(
    client: httpx.Client,
    api_key: str,
    model: str,
    transcript: str,
    provider: str = "openrouter",
    *,
    meeting_ref: str,
    language: str = "en",
    duration_sec: float | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = 2,
    verbose: bool = False,
) -> RequestResult:
    payload = {
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    meeting_ref=meeting_ref or "unknown",
                    language=language or "en",
                    duration_sec=duration_sec or 0.0,
                    transcript=(transcript or "")[:8000],
                ),
            },
        ],
    }

    for attempt in range(retries + 1):
        started = time.time()
        try:
            resp = _post_request(client, provider, api_key, payload)
        except Exception as exc:
            if attempt >= retries:
                return RequestResult(status="error", error=str(exc))
            time.sleep(1.0 + attempt)
            continue

        latency = time.time() - started

        # =============================================================================
        # PATCH 4: replace ONLY the 429 block inside request_tasks()
        # =============================================================================
        if resp.status_code == 429:
            retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
            if attempt < retries:
                sleep_time = retry_after if retry_after is not None else (2.0 * (attempt + 1))
                if verbose:
                    print(f"  [RATE LIMIT] {model}: retrying in {sleep_time:.1f}s")
                time.sleep(sleep_time)
                continue
            return RequestResult(
                status="rate_limited", http_status=429, error="rate_limited",
                retry_after=retry_after, latency_sec=latency,
            )

        if resp.status_code == 403:
            if verbose:
                print(f"  [FORBIDDEN] {model}: access denied")
            return RequestResult(status="forbidden", http_status=403, error="forbidden", latency_sec=latency)

        if resp.status_code == 408 or resp.status_code >= 500:
            if attempt < retries:
                sleep_time = _parse_retry_after(resp.headers.get("Retry-After")) or (1.5 * (attempt + 1))
                if verbose:
                    print(f"  [SERVER ERROR {resp.status_code}] {model}: retrying in {sleep_time:.1f}s")
                time.sleep(sleep_time)
                continue
            return RequestResult(
                status="error", http_status=resp.status_code, error=f"http_{resp.status_code}", latency_sec=latency,
            )

        if resp.status_code != 200:
            err_preview = resp.text[:200].replace("\n", "  ")
            if verbose:
                print(f"  [ERROR {resp.status_code}] {model}: {err_preview}")
            return RequestResult(status="error", http_status=resp.status_code, error=err_preview, latency_sec=latency)

        try:
            data = resp.json()
        except Exception as exc:
            return RequestResult(status="error", http_status=200, error=f"bad_json:{exc}", latency_sec=latency)

        response_model = data.get("model") or model
        raw = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        tasks, parse_stage = parse_tasks_from_raw(raw)

        if verbose:
            preview = normalize_space(raw)[:150].replace("\n", "  ")
            print(f"  [OK] {model}: latency={latency:.2f}s, tasks={len(tasks)}, preview: {preview}...")

        # =============================================================================
        # PATCH 5: replace ONLY the final RequestResult(...) block in request_tasks()
        # =============================================================================
        return RequestResult(
            status="ok",
            http_status=200,
            response_model=response_model,
            raw=raw,
            raw_preview=normalize_space(raw)[:1000],
            parse_stage=parse_stage,
            tasks=tasks,
            latency_sec=latency,
        )

    return RequestResult(status="error", error="unknown_error")

# =============================================================================
# NVIDIA Model Discovery
# =============================================================================
UNSUPPORTED_MODELS = {
    "nvidia/synthetic-video-detector", "nvidia/active-speaker-detection", "nvidia/gliner-pii",
    "nvidia/cosmos-transfer2.5-2b", "nvidia/cosmos-transfer1-7b", "nvidia/cosmos-predict1-5b",
    "nvidia/streampetr", "nvidia/sparsedrive", "nvidia/bevformer", "nvidia/usdcode",
    "nvidia/usdvalidate", "nvidia/studiovoice", "nvidia/nv-embedcode-7b-v1",
    "nvidia/nv-embed-v1", "nvidia/rerank-qa-mistral-4b", "nvidia/llama-3_2-nemoretriever-300m-embed-v1",
    "meta/esm2-650m", "meta/esmfold", "google/paligemma", "nvidia/magpie-tts-zeroshot",
    "nvidia/nemotron-3-content-safety", "nvidia/nemotron-content-safety-reasoning-4b",
    "nvidia/llama-3.1-nemotron-safety-guard-8b-v3", "meta/llama-guard-4-12b",
}

# =============================================================================
# PATCH 8: OPTIONAL but strongly recommended
# Replace NVIDIA free model list with better task-extraction subset
# =============================================================================
def get_nvidia_free_models(skip_unsupported: bool = True) -> list[str]:
    known_free = [
        # BEST overall instruction following
        "meta/llama-4-maverick-17b-128e-instruct",
        # Strong structured extraction
        "mistralai/mistral-large-3-675b-instruct-2512",
        "mistralai/mistral-nemotron",
        # Surprisingly strong for extraction
        "google/gemma-3n-e4b-it",
        # Fast smaller models
        "google/gemma-3n-e2b-it",
        "nvidia/nemotron-mini-4b-instruct",
        # Coding-oriented but decent structured output
        "qwen/qwen3-coder-480b-a35b-instruct",
        # Experimental
        "upstage/solar-10.7b-instruct",
        "bytedance/seed-oss-36b-instruct",
    ]
    if skip_unsupported:
        return [m for m in known_free if m not in UNSUPPORTED_MODELS]
    return known_free

# =============================================================================
# Benchmarking Logic
# =============================================================================
def summarize_numeric(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None

def fmt_float(value: float | None, digits: int = 4) -> float | None:
    return round(float(value), digits) if value is not None else None

def benchmark_model(
    client: httpx.Client,
    api_key: str,
    model: str,
    records: list[dict[str, Any]],
    provider: str = "openrouter",
    *,
    delay_sec: float,
    preflight: bool = True,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = 2,
    verbose: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    if preflight:
        probe = probe_model(client, api_key, model, provider=provider, verbose=verbose)
        if probe.status in {"rate_limited", "forbidden", "error"}:
            if verbose:
                print(f"  [PREFLIGHT FAILED] {model}: {probe.status} - {probe.error}")
            return (
                {
                    "model": model, "response_model": None, "status": probe.status,
                    "n_total": len(records), "n_ok": 0, "n_failed": 0,
                    "n_rate_limited": 1 if probe.status == "rate_limited" else 0,
                    "n_forbidden": 1 if probe.status == "forbidden" else 0,
                    "n_parse_failed": 0, "avg_task_f1": None, "avg_precision": None,
                    "avg_recall": None, "avg_assignee_accuracy": None, "avg_deadline_accuracy": None,
                    "avg_hallucination_rate": None, "avg_predicted_tasks": None,
                    "avg_matched_tasks": None, "avg_latency_sec": None, "avg_raw_chars": None,
                    "notes": probe.error or "preflight_failed",
                },
                [],
            )

    response_models: list[str] = []
    metrics_acc = defaultdict(list)
    total = len(records)

    for idx, rec in enumerate(records, start=1):
        transcript = rec.get("transcript", "") or ""
        gold_tasks = rec.get("tasks", []) or []
        meeting_ref = rec.get("meeting_ref") or rec.get("id") or rec.get("_source_file") or f"row_{idx}"
        duration_sec = rec.get("duration_sec")
        language = rec.get("language", "en")

        if verbose or idx == 1 or idx % 5 == 0:
            print(f"  [{idx}/{total}] {meeting_ref} ... ", end="", flush=True)

        time.sleep(max(0.0, delay_sec))

        rr = request_tasks(
            client, api_key, model, transcript, provider=provider,
            meeting_ref=str(meeting_ref), language=language, duration_sec=duration_sec,
            max_tokens=max_tokens, retries=retries, verbose=verbose,
        )

        response_models.append(rr.response_model or model)
        effective_latency = rr.latency_sec if rr.latency_sec is not None else 0.0

        row = {
            "model": model, "response_model": rr.response_model or model,
            "meeting_ref": str(meeting_ref), "source_file": rec.get("_source_file"),
            "status": rr.status, "http_status": rr.http_status, "error": rr.error,
            "parse_stage": rr.parse_stage, "predicted_tasks": len(rr.tasks),
            "gold_tasks": len(gold_tasks), "raw_preview": rr.raw_preview,
            "latency_sec": fmt_float(effective_latency, 3), "raw_chars": len(rr.raw or ""),
        }

        # =============================================================================
        # PATCH 6: inside benchmark_model(), replace this block:
        # =============================================================================
        if rr.status == "ok":
            task_metrics = evaluate_tasks(rr.tasks, gold_tasks)
            row.update(task_metrics)

            if not rr.tasks:
                row["note"] = "empty_output"

            for key in (
                "task_set_f1", "task_set_precision", "task_set_recall",
                "assignee_accuracy", "deadline_accuracy", "hallucination_rate",
                "predicted_tasks", "matched_tasks",
            ):
                metrics_acc[key].append(task_metrics[key])

            metrics_acc["latency_sec"].append(effective_latency)
            metrics_acc["raw_chars"].append(len(rr.raw or ""))

            if verbose or idx == 1 or idx % 5 == 0:
                print(f" ✓ f1={task_metrics['task_set_f1']:.3f} tasks={len(rr.tasks)}")

        elif rr.status == "rate_limited":
            row["note"] = "stopped_after_rate_limit"
            rows.append(row)
            if verbose or idx == 1 or idx % 5 == 0:
                print(" ✗ RATE LIMITED")
            break
        else:
            if verbose or idx == 1 or idx % 5 == 0:
                err_short = (rr.error or "no error")[:50]
                print(f" ✗ {rr.status} ({err_short})")

        rows.append(row)

    ok_rows = [r for r in rows if r.get("status") == "ok"]

    if ok_rows:
        agg_status = "ok"
    elif any(r.get("status") == "rate_limited" for r in rows):
        agg_status = "rate_limited"
    elif any(r.get("status") == "forbidden" for r in rows):
        agg_status = "forbidden"
    else:
        agg_status = "failed"

    summary = {
        "model": model,
        "response_model": response_models[-1] if response_models else None,
        "status": agg_status,
        "n_total": len(records),
        "n_ok": len(ok_rows),
        "n_failed": sum(1 for r in rows if r.get("status") == "error"),
        "n_rate_limited": sum(1 for r in rows if r.get("status") == "rate_limited"),
        "n_forbidden": sum(1 for r in rows if r.get("status") == "forbidden"),
        "n_parse_failed": sum(1 for r in rows if r.get("status") == "parse_failed"),
        "avg_task_f1": fmt_float(summarize_numeric(metrics_acc["task_set_f1"]), 4),
        "avg_precision": fmt_float(summarize_numeric(metrics_acc["task_set_precision"]), 4),
        "avg_recall": fmt_float(summarize_numeric(metrics_acc["task_set_recall"]), 4),
        "avg_assignee_accuracy": fmt_float(
            summarize_numeric([v for v in metrics_acc["assignee_accuracy"] if v is not None]), 4
        ),
        "avg_deadline_accuracy": fmt_float(
            summarize_numeric([v for v in metrics_acc["deadline_accuracy"] if v is not None]), 4
        ),
        "avg_hallucination_rate": fmt_float(summarize_numeric(metrics_acc["hallucination_rate"]), 4),
        "avg_predicted_tasks": fmt_float(summarize_numeric(metrics_acc["predicted_tasks"]), 4),
        "avg_matched_tasks": fmt_float(summarize_numeric(metrics_acc["matched_tasks"]), 4),
        "avg_latency_sec": fmt_float(summarize_numeric(metrics_acc["latency_sec"]), 3),
        "avg_raw_chars": fmt_float(summarize_numeric(metrics_acc["raw_chars"]), 1),
        "notes": None,
    }

    if summary["n_rate_limited"]:
        summary["notes"] = "rate_limited"
    elif summary["n_forbidden"]:
        summary["notes"] = "forbidden"
    elif summary["n_parse_failed"] and not ok_rows:
        summary["notes"] = "all_parse_failed"

    return summary, rows

def print_summary_table(summary: list[dict[str, Any]]) -> None:
    if not summary:
        print("No results.")
        return
    headers = [
        "model", "status", "n_ok", "n_total", "avg_f1", "avg_prec", "avg_rec",
        "avg_assignee", "avg_deadline", "avg_latency", "rate_limited", "forbidden",
    ]
    widths = {h: len(h) for h in headers}
    rows: list[dict[str, str]] = []

    for s in summary:
        row = {
            "model": s["model"], "status": s["status"], "n_ok": str(s["n_ok"]),
            "n_total": str(s["n_total"]),
            "avg_f1": "" if s["avg_task_f1"] is None else f'{s["avg_task_f1"]:.4f}',
            "avg_prec": "" if s["avg_precision"] is None else f'{s["avg_precision"]:.4f}',
            "avg_rec": "" if s["avg_recall"] is None else f'{s["avg_recall"]:.4f}',
            "avg_assignee": "" if s["avg_assignee_accuracy"] is None else f'{s["avg_assignee_accuracy"]:.4f}',
            "avg_deadline": "" if s["avg_deadline_accuracy"] is None else f'{s["avg_deadline_accuracy"]:.4f}',
            "avg_latency": "" if s["avg_latency_sec"] is None else f'{s["avg_latency_sec"]:.2f}s',
            "rate_limited": str(s["n_rate_limited"]), "forbidden": str(s["n_forbidden"]),
        }
        rows.append(row)
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))

    def fmt_row(row: dict[str, str]) -> str:
        return " | ".join(row[h].ljust(widths[h]) for h in headers)

    print(fmt_row({h: h for h in headers}))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(fmt_row(row))

# =============================================================================
# Entry Point
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare task extraction models on gold annotations.")
    parser.add_argument("--gold-dir", required=True, help="Folder with gold JSON files")
    parser.add_argument("--models", required=False, help="Comma-separated model slugs")
    parser.add_argument("--out-csv", required=True, help="Output CSV summary path")
    parser.add_argument("--out-json", required=True, help="Output JSON report path")
    parser.add_argument("--delay-sec", type=float, default=None, help="Delay between API requests (auto for NVIDIA if None)")
    parser.add_argument("--retries", type=int, default=2, help="Retries for transient errors")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--no-preflight", action="store_true", help="Disable preflight check")
    parser.add_argument("--provider", choices=["openrouter", "nvidia"], default="openrouter")
    parser.add_argument("--list-nvidia-free", action="store_true", help="List known free NVIDIA models")
    parser.add_argument("--include-unsupported", action="store_true", help="Include non-text-generation models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.list_nvidia_free:
        models = get_nvidia_free_models(skip_unsupported=not args.include_unsupported)
        print("Free NVIDIA models (text-generation):")
        for m in models:
            print(f"  - {m}")
        if args.include_unsupported:
            print("\n⚠️  Including unsupported models (may fail):")
            for m in UNSUPPORTED_MODELS:
                print(f"  - {m}")
        return

    api_key = os.getenv("NVIDIA_API_KEY" if args.provider == "nvidia" else "OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(f"{'NVIDIA' if args.provider == 'nvidia' else 'OpenRouter'}_API_KEY is not set")

    gold_dir = Path(args.gold_dir).expanduser().resolve()
    records = parse_gold_dir(gold_dir)
    if not records:
        raise SystemExit(f"No JSON files found in {gold_dir}")

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.provider == "nvidia":
        models = get_nvidia_free_models(skip_unsupported=not args.include_unsupported)
        print(f"Using {len(models)} NVIDIA models (text-generation only) " + (" + unsupported " if args.include_unsupported else "") + ".")
    else:
        raise SystemExit("No models provided. Use --models or --provider nvidia")

    if not models:
        raise SystemExit("No models provided")

    delay_sec = args.delay_sec if args.delay_sec is not None else (2.0 if args.provider == "nvidia" else 3.0)
    if args.provider == "nvidia" and args.verbose:
        print(f"Using NVIDIA rate limit delay: {delay_sec}s (40 RPM limit)")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    detailed_runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    timeout = httpx.Timeout(DEFAULT_TIMEOUT, connect=20.0)
    limits = httpx.Limits(max_keepalive_connections=2, max_connections=4)

    with httpx.Client(timeout=timeout, limits=limits) as client:
        for m_idx, model in enumerate(models, start=1):
            print(f"\n[{m_idx}/{len(models)}] === {model} ({args.provider}) ===")
            summary, runs = benchmark_model(
                client, api_key, model, records, provider=args.provider,
                delay_sec=delay_sec, preflight=not args.no_preflight,
                max_tokens=args.max_tokens, retries=args.retries, verbose=args.verbose,
            )
            summaries.append(summary)
            detailed_runs.extend(runs)

            if summary["status"] in {"rate_limited", "forbidden"}:
                print(f"[SKIP] {model}: {summary['status']}")
            else:
                lat = summary["avg_latency_sec"]
                lat_str = f"{lat}s" if lat is not None else "N/A"
                print(
                    f"[DONE] {model}: ok={summary['n_ok']}/{summary['n_total']}  "
                    f"f1={summary['avg_task_f1']} prec={summary['avg_precision']} rec={summary['avg_recall']}  "
                    f"latency={lat_str}"
                )
            time.sleep(max(0.0, delay_sec))

    summaries_sorted = sorted(
        summaries,
        key=lambda s: (s["avg_task_f1"] or 0.0, s["avg_precision"] or 0.0, s["avg_recall"] or 0.0),
        reverse=True,
    )

    report = {
        "generated_at": started_at,
        "gold_dir": str(gold_dir),
        "records": len(records),
        "provider": args.provider,
        "models_requested": models,
        "settings": {
            "delay_sec": delay_sec, "retries": args.retries,
            "max_tokens": args.max_tokens, "preflight": not args.no_preflight,
        },
        "summary": summaries_sorted,
        "runs": detailed_runs,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model", "status", "n_ok", "n_total", "avg_task_f1", "avg_precision", "avg_recall",
            "avg_assignee_accuracy", "avg_deadline_accuracy", "avg_hallucination_rate",
            "avg_predicted_tasks", "avg_matched_tasks", "avg_latency_sec", "avg_raw_chars",
            "n_rate_limited", "n_forbidden", "n_parse_failed", "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries_sorted:
            writer.writerow({k: s.get(k) for k in fieldnames})

    print("\n=== FINAL SUMMARY ===")
    print_summary_table(summaries_sorted)
    print(f"\nCSV: {out_csv}\nJSON: {out_json}")

if __name__ == "__main__":
    main()