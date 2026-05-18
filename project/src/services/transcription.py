"""Transcription service with audio preprocessing, Whisper ASR and diarization.

Enhancements over the baseline:
- normalized per-segment confidence
- diarization labels kept in segments
- a speaker-labeled transcript is generated for downstream task extraction
- lightweight speaker alias inference from self-introductions
"""
from __future__ import annotations

import logging
import math
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Union

import torch
import whisper

from .audio_preprocessing import preprocess_audio
from .diarization import diarize_audio
from ..config import settings

logger = logging.getLogger(__name__)

_model = None
_model_device: Optional[str] = None
_model_dtype = None


def get_model():
    """Lazy-load Whisper model once per process."""
    global _model, _model_device, _model_dtype

    if _model is not None:
        return _model

    device = "cuda" if settings.WHISPER_DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
    _model_device = device

    logger.info("[TRANSCRIPTION] Loading Whisper model %s on %s", settings.WHISPER_MODEL, device)
    _model = whisper.load_model(settings.WHISPER_MODEL, device=device)

    try:
        _model_dtype = _model.model.dtype if hasattr(_model, "model") else torch.float32
    except Exception:
        _model_dtype = torch.float32

    logger.info("[TRANSCRIPTION] Model loaded - device=%s, dtype=%s", device, _model_dtype)
    return _model


def _segment_confidence(seg: Dict[str, Any]) -> Optional[float]:
    """Convert Whisper's avg_logprob to a soft [0..1] score."""
    for key in ("confidence", "avg_logprob"):
        value = seg.get(key)
        if isinstance(value, (int, float)):
            if key == "confidence":
                return float(max(0.0, min(1.0, value)))
            return float(max(0.0, min(1.0, math.exp(value))))
    return None


def _normalize_speaker_label(label: Any) -> str:
    text = str(label or "Unknown").strip()
    m = re.search(r"(\d+)", text)
    if text.upper().startswith("SPEAKER") and m:
        return f"SPEAKER_{int(m.group(1)):02d}"
    if text.lower().startswith("speaker") and m:
        return f"SPEAKER_{int(m.group(1)):02d}"
    return text or "Unknown"


def merge_speakers(transcript_segments: List[Dict[str, Any]], speaker_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge Whisper segments with diarization output using overlap matching."""
    if not speaker_segments:
        return transcript_segments

    merged: List[Dict[str, Any]] = []
    for ts in transcript_segments:
        start = float(ts["start"])
        end = float(ts["end"])
        best_speaker = None
        best_overlap = 0.0
        mid = (start + end) / 2

        for sp in speaker_segments:
            s_start = float(sp.get("start", 0.0))
            s_end = float(sp.get("end", 0.0))
            overlap = max(0.0, min(end, s_end) - max(start, s_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sp.get("speaker")

        if best_speaker is None and speaker_segments:
            min_dist = float("inf")
            for sp in speaker_segments:
                sp_mid = (float(sp.get("start", 0.0)) + float(sp.get("end", 0.0))) / 2
                dist = abs(mid - sp_mid)
                if dist < min_dist:
                    min_dist = dist
                    best_speaker = sp.get("speaker")

        ts_copy = dict(ts)
        ts_copy["speaker"] = _normalize_speaker_label(best_speaker if best_speaker is not None else "Unknown")
        merged.append(ts_copy)

    return merged


def _looks_like_person_name(candidate: str) -> bool:
    candidate = (candidate or "").strip()
    if not candidate:
        return False

    # Only accept explicit-looking names, not ordinary phrases.
    if not re.fullmatch(r"[A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){0,2}", candidate):
        return False

    bad_tokens = {
        "getting", "going", "doing", "waking", "waking up", "shadow", "everything",
        "problem", "meeting", "agenda", "project", "team", "design", "function",
        "control", "system", "pain", "hurt", "this", "that", "there", "here",
    }
    low = candidate.lower()
    return not any(tok in low for tok in bad_tokens)


_ROSTER_STOP = {
    "the", "this", "that", "we", "i", "he", "she", "it", "a", "an",
    "ok", "okay", "right", "there", "here", "so", "well", "also", "then",
    "now", "just", "and", "but", "or", "in", "at", "by", "to", "you",
    "you're", "i'm", "they", "them", "their", "our", "your", "my",
    "next", "last", "first", "are", "is",
}

_ROLE_KW_RE = re.compile(
    r"industrial designer|marketing expert|user interface designer"
    r"|project manager|ux designer|ui designer|product manager"
    r"|software engineer|developer|designer|researcher",
    re.IGNORECASE,
)


def _is_real_name(text: str) -> bool:
    words = text.strip().split()
    if not words or len(words) > 4:
        return False
    if not all(w and w[0].isupper() for w in words):
        return False
    if any(w.lower() in _ROSTER_STOP for w in words):
        return False
    if any(len(w) < 2 for w in words):
        return False
    return True


def _extract_roster_from_text(text: str) -> list[dict]:
    """
    Extract ALL named participants and their roles from raw transcript text.
    Handles both self-introductions and host-led introductions like:
      "we have Ebenezer Ademisoy... your role is? I'm the marketing expert"
      "my name's Adam Duggard"
      "Tarik Rahman ... Industrial designer"
      "lastly we have Dave Cochran ... user interface designer"
    Returns list of {name, role} dicts (deduped).
    """
    roster: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, role: Optional[str]):
        name = name.strip()
        if not _is_real_name(name):
            return
        key = name.lower()
        if key in seen:
            return
        seen.add(key)
        roster.append({"name": name, "role": (role or "").strip() or None})

    # Pattern 1: self-intro
    for m in re.finditer(
        r"(?:my name(?:'s| is)|i(?:'m| am))\s+([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)?)",
        text, re.IGNORECASE,
    ):
        _add(m.group(1), None)


    # Pattern 2: host-led "we have <Name>" — sentence-aware look-ahead for role
    # Pattern 2: host-led 'we have <Name>' — sentence-aware role look-ahead
    # Split on both punctuation and newlines
    sentences = re.split(r'(?:[.!?]\s+|\n+)', text)
    # Compile triggers with IGNORECASE; name pattern WITHOUT so [A-Z] = strictly uppercase
    _INTRO_TRIGGER = re.compile(
        r'we have|next we have|lastly we have|introducing|and then we have',
        re.IGNORECASE,
    )
    _NAME_CAP_RE = re.compile(r'([A-Z][a-z\'\-]+(?:\s+[A-Z][a-z\'\-]+)?)')
    for i, sent in enumerate(sentences):
        trig = _INTRO_TRIGGER.search(sent)
        if not trig:
            continue
        # Look for name in current sentence + next 2 sentences
        # (handles 'Next we have?' where name is in next sentence)
        search_ctx = sent[trig.end():] + ' ' + ' '.join(sentences[i+1:i+3])
        name_m = _NAME_CAP_RE.search(search_ctx[:150])
        if not name_m:
            continue
        candidate = name_m.group(1).strip()
        if not _is_real_name(candidate):
            continue
        context = ' '.join(sentences[i:i+6])
        role_m = _ROLE_KW_RE.search(context)
        _add(candidate, role_m.group(0) if role_m else None)

    # Pattern 3: fill in missing roles for already-found names
    for entry in roster:
        if entry["role"] is None:
            idx = text.lower().find(entry["name"].lower())
            if idx >= 0:
                role_m = _ROLE_KW_RE.search(text[idx: idx + 300])
                if role_m:
                    entry["role"] = role_m.group(0).strip()

    return roster


def _infer_speaker_aliases(segments: List[Dict[str, Any]]) -> dict[str, str]:
    """
    Infer SPEAKER_XX → name mapping from diarized segments.
    Uses self-introductions AND host-led introductions.
    Each SPEAKER_XX is only assigned one name (first confident match wins).
    """
    aliases: dict[str, str] = {}

    self_intro_patterns = [
        r"(?:i'm|i am|my name is|this is|name's)\s+([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)?)",
    ]

    for seg in segments[:60]:  # check more segments (was 40)
        speaker = _normalize_speaker_label(seg.get("speaker") or "Unknown")
        text = str(seg.get("text") or "")
        if speaker in aliases:
            continue

        for pat in self_intro_patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if not m:
                continue
            candidate = m.group(1).strip()
            if _looks_like_person_name(candidate) and len(candidate.split()) <= 3:
                aliases[speaker] = candidate
                break

    return aliases


def _build_roster_from_transcript(full_text: str) -> list[dict]:
    """
    Public helper: extract all named participants + roles from the full transcript.
    Used downstream for role-based task assignment.
    """
    return _extract_roster_from_text(full_text)


def _apply_speaker_aliases(segments: List[Dict[str, Any]], alias_map: dict[str, str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seg in segments:
        speaker = _normalize_speaker_label(seg.get("speaker") or "Unknown")
        alias = alias_map.get(speaker)

        item = dict(seg)
        item["speaker_label"] = speaker
        item["speaker_name"] = alias
        item["speaker_display"] = alias or speaker
        out.append(item)
    return out


def _segments_to_speaker_transcript(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    last_speaker: Optional[str] = None

    for seg in segments:
        speaker = str(seg.get("speaker_display") or seg.get("speaker_name") or seg.get("speaker") or "Unknown").strip()
        text = str(seg.get("text") or "").strip()
        if not text:
            continue

        line = f"{speaker}: {text}"
        if speaker == last_speaker and lines:
            lines[-1] = f"{lines[-1]} {text}"
        else:
            lines.append(line)
            last_speaker = speaker

    return "\n".join(lines).strip()


def transcribe_from_bytes(audio_source: Union[bytes, str], filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe audio with preprocessing and optional diarization.
    """
    temp_wav: Optional[str] = None

    try:
        try:
            temp_wav = preprocess_audio(audio_source, filename)
        except Exception as e:
            logger.warning("Preprocessing failed: %s; saving raw audio to temp file", e)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            if isinstance(audio_source, bytes):
                tmp.write(audio_source)
            else:
                with open(audio_source, "rb") as src:
                    tmp.write(src.read())
            tmp.close()
            temp_wav = tmp.name

        model = get_model()
        use_fp16 = _model_device == "cuda" and _model_dtype == torch.float16

        result = model.transcribe(
            audio=temp_wav,
            task="transcribe",
            word_timestamps=True,
            temperature=0.0,
            verbose=False,
            fp16=use_fp16,
        )

        whisper_segments: List[Dict[str, Any]] = []
        confidences: List[float] = []

        for seg in result.get("segments", []):
            segment_data: Dict[str, Any] = {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": str(seg["text"]).strip(),
            }
            conf = _segment_confidence(seg)
            if conf is not None:
                segment_data["confidence"] = conf
                confidences.append(conf)
            whisper_segments.append(segment_data)

        merged_segments = whisper_segments
        has_diarization = False

        if settings.DIARIZATION_ENABLED and temp_wav and os.path.exists(temp_wav):
            try:
                logger.info("[TRANSCRIPTION] Starting diarization...")
                speaker_segments = diarize_audio(temp_wav)
                if speaker_segments:
                    merged_segments = merge_speakers(whisper_segments, speaker_segments)
                    has_diarization = True
                    logger.info("[TRANSCRIPTION] Diarization complete: %d speaker segments", len(speaker_segments))
                else:
                    logger.warning("[TRANSCRIPTION] Diarization returned no segments")
            except Exception as e:
                logger.error("[TRANSCRIPTION] Diarization failed: %s", e, exc_info=True)

        speaker_aliases = _infer_speaker_aliases(merged_segments)
        enriched_segments = _apply_speaker_aliases(merged_segments, speaker_aliases)
        speaker_transcript = _segments_to_speaker_transcript(enriched_segments) or str(result.get("text", "")).strip()

        # Extract full participant roster from transcript (names + roles mentioned by anyone)
        full_text = str(result.get("text", "")).strip()
        meeting_roster = _build_roster_from_transcript(full_text) if full_text else []

        confidence = float(sum(confidences) / len(confidences)) if confidences else None

        output: Dict[str, Any] = {
            "text": str(result.get("text", "")).strip(),
            "language": result.get("language") or "en",
            "segments": enriched_segments,
            "speaker_transcript": speaker_transcript,
            "speaker_aliases": speaker_aliases,
            "meeting_roster": meeting_roster,
            "confidence": confidence,
            "has_diarization": has_diarization,
        }

        logger.info(
            "[TRANSCRIPTION] Completed: %d segments, language=%s, confidence=%s, diarization=%s",
            len(enriched_segments),
            output["language"],
            output["confidence"],
            output["has_diarization"],
        )

        return output

    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.unlink(temp_wav)
            except OSError as e:
                logger.warning("Failed to delete temp file %s: %s", temp_wav, e)