"""Task assignment engine.

Strategy order:
1. Resolve speaker_hint using meeting-level speaker aliases.
2. Match assignee_hint to participant name/email/role.
3. [NEW] If hint present but no participant found → use raw hint as assignee.
4. Infer role from task text and match participant.role.
5. Extract a concrete name from the description/snippet.
6. Apply explicit regex/role rules.
7. [NEW] Assign by SPEAKER_XX label when no other info available.
8. Round-robin fallback only when participants exist in DB.
9. Keep assignee None only when there is genuinely nothing to go on.

The engine enriches each task with:
    assignee, assignee_source, assignment_confidence, speaker_resolved
"""
from __future__ import annotations

import json
import re
from difflib import get_close_matches
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ..models import Participant, Rule


def load_participants(session: Session) -> List[Participant]:
    return session.exec(select(Participant)).all()


def load_rules(session: Session) -> List[Rule]:
    return session.exec(select(Rule).order_by(Rule.priority)).all()


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _tags_as_text(tags: Optional[str]) -> str:
    if not tags:
        return ""
    try:
        parsed = json.loads(tags)
        if isinstance(parsed, dict):
            return _norm(" ".join(f"{k}:{v}" for k, v in parsed.items()))
        if isinstance(parsed, list):
            return _norm(" ".join(str(x) for x in parsed))
    except Exception:
        pass
    return _norm(tags.replace(",", " "))


def _participant_blob(p: Participant) -> str:
    return _norm(" ".join(filter(None, [p.name, p.email, p.role, _tags_as_text(p.tags)])))


def _match_name(participants: List[Participant], name_hint: Optional[str]) -> Optional[Participant]:
    if not name_hint:
        return None
    hint = _norm(name_hint)
    if not hint:
        return None

    for p in participants:
        blob = _participant_blob(p)
        if hint in blob:
            return p

    names = [p.name for p in participants if p.name]
    close = get_close_matches(name_hint, names, n=1, cutoff=0.72)
    if close:
        for p in participants:
            if p.name == close[0]:
                return p

    return None


def _match_role(participants: List[Participant], role_hint: Optional[str]) -> Optional[Participant]:
    if not role_hint:
        return None
    hint = _norm(role_hint)
    if not hint:
        return None

    for p in participants:
        role = _norm(p.role)
        if role and (hint in role or role in hint):
            return p
    return None


def _speaker_alias_map(meeting_info: Optional[dict]) -> dict[str, str]:
    if not meeting_info:
        return {}

    aliases: dict[str, str] = {}
    for key in ("speaker_aliases", "speaker_aliases_manual", "speaker_name_map"):
        value = meeting_info.get(key) or {}
        if isinstance(value, dict):
            for k, v in value.items():
                if v:
                    aliases[str(k)] = str(v)
    return aliases


def _resolve_speaker_hint(task: dict, meeting_info: Optional[dict], participants: List[Participant]) -> Optional[str]:
    speaker_hint = (
        task.get("speaker_hint")
        or task.get("speaker")
        or task.get("speaker_label")
        or task.get("speaker_name")
        or task.get("speaker_display")
    )
    if not speaker_hint:
        return None

    speaker_hint_str = str(speaker_hint).strip()
    if not speaker_hint_str:
        return None

    aliases = _speaker_alias_map(meeting_info)
    resolved = aliases.get(speaker_hint_str)

    if not resolved:
        digits = re.search(r"(\d+)", speaker_hint_str)
        if digits:
            idx = digits.group(1)
            for key, value in aliases.items():
                if re.search(rf"\b{re.escape(idx)}\b", key) or re.search(rf"\b{re.escape(idx)}\b", value):
                    resolved = value
                    break

    if resolved:
        return resolved

    if _match_name(participants, speaker_hint_str):
        return speaker_hint_str

    # Keep SPEAKER_XX / single-letter labels as labels so tasks can be grouped by speaker
    # even without a named alias
    if re.fullmatch(r"(?:SPEAKER_\d+|Speaker\s+\d+|[A-Z])", speaker_hint_str):
        return speaker_hint_str

    return None


_ROLE_KEYWORDS = {
    "project manager": ["project manager", "pm", "manager"],
    "industrial designer": ["industrial designer", "designer", "design"],
    "marketing expert": ["marketing expert", "marketing", "market"],
    "user interface": ["user interface", "ui", "interface"],
    "technical": ["technical", "tech", "engineer"],
    "facilitator": ["facilitator", "moderator", "host"],
}


def _infer_role_hint(text: str) -> Optional[str]:
    s = _norm(text)
    if not s:
        return None
    for role, keys in _ROLE_KEYWORDS.items():
        if any(k in s for k in keys):
            return role
    return None


_NAME_PATTERN = re.compile(
    r"(?:"
    r"(?:for|to|assigned to|by|from)\s+"
    r"|(?:speaking|speaker)\s+"
    r")?([A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+){0,2})"
    r"|@([\w.\-]+)"
)

# Words that must never be treated as a person name
_NOT_A_NAME: set[str] = {
    # Pronouns
    "we", "i", "he", "she", "they", "it", "you", "your", "my", "our", "their",
    "the", "this", "that", "these", "those", "me", "him", "her", "us", "them",
    "who", "what", "which", "where", "when", "all", "each", "every", "some",
    "any", "both", "few", "many", "much", "such",
    # Common English verbs (imperative form — often starts a task description)
    "summarize", "determine", "conduct", "ensure", "create", "review", "write",
    "check", "update", "send", "make", "get", "take", "have", "give", "find",
    "identify", "define", "develop", "implement", "prepare", "present",
    "analyze", "discuss", "decide", "resolve", "complete", "finish", "start",
    "begin", "coordinate", "schedule", "organize", "research", "explore",
    "evaluate", "assess", "establish", "provide", "support", "manage", "plan",
    "follow", "confirm", "contact", "draft", "collect", "share", "build",
    "test", "deploy", "configure", "document", "fix", "address", "handle",
    "look", "set", "go", "do", "try", "ask", "tell", "show", "let", "need",
    # Generic meeting words
    "meeting", "agenda", "task", "please", "should", "must", "will", "would",
    "could", "there", "here", "going", "getting", "doing", "working", "team",
    "action", "item", "next", "steps", "follow", "up", "based", "yes", "no",
    # Other common words that can start with capital
    "according", "after", "before", "during", "however", "also", "just",
    "then", "so", "and", "but", "or", "if", "when", "while", "as", "at",
    "by", "for", "from", "in", "into", "of", "on", "to", "with",
}


def _is_plausible_name(text: str) -> bool:
    """
    Return True only if text looks like a real person name.
    Rules:
    - 2+ words (FirstName LastName), each starting with capital
    - OR exactly 1 word that is not in _NOT_A_NAME and not a verb/pronoun
    - Must be ≥ 2 characters
    """
    words = text.strip().split()
    if not words:
        return False
    if any(w.lower() in _NOT_A_NAME for w in words):
        return False
    if not all(w and w[0].isupper() for w in words):
        return False
    # Single-word "names" are only accepted from explicit snippet patterns (Name: text)
    # When called from free-text extraction, require 2+ words to avoid false positives
    return True  # caller decides whether single-word is ok


def _extract_name_from_snippet(snippet: str) -> Optional[str]:
    """
    Extract a person name from a source snippet.
    Accepts:
      - "Adam Duggard: text"   → "Adam Duggard"
      - "Adam: text"           → "Adam"
      - "SPEAKER_01: text"     → "SPEAKER_01"
    """
    if not snippet:
        return None
    # SPEAKER_XX label in transcript
    m = re.search(r"\b(SPEAKER_\d+)\s*:", snippet)
    if m:
        return m.group(1)
    # Name (one or two words) followed by colon — transcript speaker label
    m = re.search(
        r"([A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)?)\s*:",
        snippet,
    )
    if m:
        candidate = m.group(1).strip()
        if candidate.lower() not in _NOT_A_NAME and len(candidate) >= 2:
            return candidate
    return None


def _extract_name_from_text(text: str) -> Optional[str]:
    """
    Extract a person name from free text (description or combined text).
    Requires at least First + Last (2 words) — single words are too ambiguous
    without a colon anchor. Use _extract_name_from_snippet for that.
    """
    for m in _NAME_PATTERN.finditer(text):
        candidate = (m.group(1) or m.group(2) or "").strip()
        if not candidate or len(candidate) < 3:
            continue
        # Require 2 words from free text (no colon anchor)
        if len(candidate.split()) < 2:
            continue
        if not _is_plausible_name(candidate):
            continue
        return candidate
    return None


def _roster_match_role(meeting_info: Optional[dict], role_hint: Optional[str]) -> Optional[str]:
    """
    Find a name from the transcript-extracted roster by role keyword.
    E.g. role_hint="industrial designer" → "Tarik Rahman"
    """
    if not role_hint or not meeting_info:
        return None
    roster: list = meeting_info.get("meeting_roster") or []
    hint = role_hint.strip().lower()
    for entry in roster:
        role = (entry.get("role") or "").strip().lower()
        if role and (hint in role or role in hint):
            return entry.get("name")
    return None


def _roster_all_names(meeting_info: Optional[dict]) -> list[str]:
    """Return all names from the meeting roster (for logging/debug)."""
    if not meeting_info:
        return []
    return [e.get("name", "") for e in (meeting_info.get("meeting_roster") or []) if e.get("name")]


def _rule_assignee(desc: str, rules: List[Rule], participants: List[Participant]) -> tuple[Optional[str], Optional[str]]:
    for r in rules:
        if r.kind == "regex" and r.pattern:
            try:
                m = re.search(r.pattern, desc, re.IGNORECASE)
            except re.error:
                continue
            if not m:
                continue

            extracted = m.group(1) if m.lastindex else m.group(0)
            extracted = re.sub(r"[^\w\s@.-]", "", str(extracted)).strip()
            if extracted:
                p = _match_name(participants, extracted)
                if p:
                    return (p.email or p.name, "rule:regex:name")
                return (extracted, "rule:regex:raw")
            return (None, "rule:regex")

        if r.kind == "role_lookup" and r.pattern:
            try:
                obj = json.loads(r.pattern)
            except json.JSONDecodeError:
                continue
            role = _norm(obj.get("role"))
            default_assignee = obj.get("assignee")
            if role and role in _norm(desc):
                return (default_assignee, "rule:role_lookup")

    return (None, None)


def assign_task_to_participant(
    task: dict,
    meeting_info: Optional[dict],
    participants: List[Participant],
    rules: List[Rule],
    round_robin_idx: int,
) -> tuple[Optional[str], str, float]:
    # ── Step 1: resolve speaker hint ────────────────────────────────────────
    speaker_resolved = _resolve_speaker_hint(task, meeting_info, participants)
    if speaker_resolved:
        p = _match_name(participants, speaker_resolved)
        if p:
            return p.name, "speaker_alias", 0.95
        # Even without a participant entry, use the resolved name/label
        # Named alias (e.g. "Adam Duggard") → good confidence
        if not re.fullmatch(r"(?:SPEAKER_\d+|Speaker\s+\d+|[A-Z])", speaker_resolved):
            return speaker_resolved, "speaker_alias_raw", 0.85

    # ── Step 2: assignee_hint ────────────────────────────────────────────────
    assignee_hint = (task.get("assignee_hint") or "").strip() or None
    if assignee_hint:
        p = _match_name(participants, assignee_hint)
        if p:
            return p.name, "assignee_hint", 0.90
        # [NEW] No participant match — still use the raw hint if it looks like a name
        words = assignee_hint.split()
        if 1 <= len(words) <= 4 and assignee_hint[0].isupper():
            return assignee_hint, "assignee_hint_raw", 0.65

    # ── Step 3: role inference from description ──────────────────────────────
    role_hint = _infer_role_hint(task.get("description", ""))
    p = _match_role(participants, role_hint)
    if p:
        return p.name, "role_hint", 0.70
    # [ROSTER] No DB participant — try meeting transcript roster
    roster_name = _roster_match_role(meeting_info, role_hint)
    if roster_name:
        return roster_name, "role_roster", 0.68

    # ── Step 4: name extraction from description + snippet ──────────────────
    combined = " ".join(filter(None, [task.get("description", ""), task.get("source_snippet", "")]))
    candidate = (
        _extract_name_from_snippet(task.get("source_snippet", ""))
        or _extract_name_from_text(combined)
    )
    if candidate:
        p = _match_name(participants, candidate)
        if p:
            return p.name, "name_in_text", 0.80
        # Use extracted name/label if it's not a SPEAKER_XX (those go to step 7)
        if not re.fullmatch(r"SPEAKER_\d+", candidate):
            return candidate, "name_in_snippet", 0.55

    # ── Step 5: role from combined text ─────────────────────────────────────
    role_text = _infer_role_hint(combined)
    p = _match_role(participants, role_text)
    if p:
        return p.name, "role_text", 0.65
    roster_name = _roster_match_role(meeting_info, role_text)
    if roster_name:
        return roster_name, "role_roster_text", 0.63

    # ── Step 6: explicit rules ───────────────────────────────────────────────
    rule_assignee, rule_source = _rule_assignee(task.get("description", ""), rules, participants)
    if rule_assignee:
        return rule_assignee, rule_source or "rule", 0.75

    # ── Step 7: [NEW] fall back to SPEAKER_XX label ──────────────────────────
    # Even without a name, group tasks by speaker for the user to rename later
    if speaker_resolved and re.fullmatch(r"(?:SPEAKER_\d+|Speaker\s+\d+|[A-Z])", speaker_resolved):
        return speaker_resolved, "speaker_label", 0.30

    # ── Step 8: round-robin (only if participants are registered) ────────────
    desc_norm = _norm(task.get("description", ""))
    if participants and len(desc_norm.split()) >= 4:
        p = participants[round_robin_idx % len(participants)]
        return p.name, "round_robin", 0.15

    return None, "unassigned", 0.0


def assign_tasks_to_participants(
    tasks: List[dict],
    session: Session,
    meeting_info: Optional[dict] = None,
) -> List[dict]:
    participants = load_participants(session)
    rules = load_rules(session)

    enriched: List[dict] = []
    rr_index = 0

    for task in tasks or []:
        assignee, source, confidence = assign_task_to_participant(task, meeting_info, participants, rules, rr_index)
        speaker_resolved = _resolve_speaker_hint(task, meeting_info, participants)

        item = dict(task)
        item["assignee"] = assignee
        item["assignee_source"] = source
        item["assignment_confidence"] = confidence
        if speaker_resolved:
            item["speaker_resolved"] = speaker_resolved

        if source in ("round_robin",) or (assignee is None and participants and len(_norm(item.get("description", "")).split()) >= 4):
            rr_index += 1

        enriched.append(item)

    return enriched
