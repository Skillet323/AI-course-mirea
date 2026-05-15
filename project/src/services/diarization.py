"""
Speaker diarization service with timeout protection.

Primary:
    pyannote/speaker-diarization-community-1
Fallback:
    resemblyzer + SpectralClustering
"""
from __future__ import annotations

import logging
import os
import re
import time
import multiprocessing as mp
from typing import Any, Dict, List, Optional

import torch
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)

# Force 'spawn' method for multiprocessing (required for CUDA + pyannote)
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

_pyannote_available: Optional[bool] = None
_pyannote_model_id: Optional[str] = None

# Timeout for pyannote inference (seconds)
PYANNOTE_TIMEOUT_SEC = 120


def _merge_adjacent_segments(segments: List[Dict[str, Any]], max_gap: float = 0.25) -> List[Dict[str, Any]]:
    """Merge consecutive segments with same speaker if gap is small."""
    if not segments:
        return []

    merged: List[Dict[str, Any]] = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and float(seg["start"]) - float(prev["end"]) <= max_gap:
            prev["end"] = max(float(prev["end"]), float(seg["end"]))
        else:
            merged.append(seg.copy())
    return merged


def _load_audio_as_dict(wav_path: str) -> dict[str, Any]:
    """Load audio into dict with correct shape for pyannote: [channels, samples]."""
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(wav_path, dtype="float32")
        
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data).permute(1, 0)
            
        return {"waveform": waveform, "sample_rate": sr}
    except Exception as exc:
        logger.error("[DIARIZATION] Failed to load audio with soundfile: %s", exc)
        raise


def _run_pyannote_in_subprocess(model_id: str, token: str, wav_path: str, 
                                 kwargs: dict, result_queue: mp.Queue):
    """
    Run pyannote inference in a clean subprocess.
    Pipeline is loaded INSIDE this function to avoid pickle/CUDA issues.
    """
    import sys
    import traceback
    
    try:
        logger.info("[DIARIZATION-SUB] Starting subprocess for %s", model_id)
        
        from pyannote.audio import Pipeline  # type: ignore
        
        logger.info("[DIARIZATION-SUB] Loading pipeline...")
        try:
            pipeline = Pipeline.from_pretrained(model_id, token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[DIARIZATION-SUB] Moving pipeline to %s", device)
        pipeline.to(torch.device(device))
        
        logger.info("[DIARIZATION-SUB] Loading audio...")
        audio_dict = _load_audio_as_dict(wav_path)
        
        logger.info("[DIARIZATION-SUB] Running inference...")
        result = pipeline(audio_dict, **kwargs)
        
        # Convert result to serializable format
        segments = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            })
        
        logger.info("[DIARIZATION-SUB] Inference complete, %d segments", len(segments))
        result_queue.put({"success": True, "segments": segments})
        
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)}\n{''.join(traceback.format_tb(exc.__traceback__))}"
        logger.error("[DIARIZATION-SUB] Error: %s", error_msg)
        result_queue.put({"success": False, "error": error_msg})


def _run_pyannote_with_timeout(model_id: str, token: str, wav_path: str, 
                                kwargs: dict, timeout_sec: int) -> List[Dict[str, Any]]:
    """Run pyannote in subprocess with timeout."""
    result_queue: mp.Queue = mp.Queue()
    
    process = mp.Process(
        target=_run_pyannote_in_subprocess,
        args=(model_id, token, wav_path, kwargs, result_queue)
    )
    
    logger.info("[DIARIZATION] Starting pyannote subprocess (timeout=%ds)", timeout_sec)
    process.start()
    process.join(timeout=timeout_sec)

    if process.is_alive():
        logger.error("[DIARIZATION] TIMEOUT after %ds — terminating subprocess", timeout_sec)
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise TimeoutError(f"pyannote inference exceeded {timeout_sec} seconds")

    if result_queue.empty():
        raise RuntimeError("pyannote subprocess ended without result")
        
    outcome = result_queue.get()
    if not outcome["success"]:
        raise RuntimeError(outcome["error"])
        
    return outcome["segments"]


def _diarize_pyannote(wav_path: str, n_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    token: str = settings.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not token:
        logger.warning("[DIARIZATION] HF_TOKEN not set — skipping pyannote")
        return None

    try:
        kwargs: Dict[str, Any] = {}
        if n_speakers is not None:
            kwargs["num_speakers"] = n_speakers

        # Try preferred models in order
        preferred_models = [
            "pyannote/speaker-diarization-community-1",
            "pyannote/speaker-diarization-3.1",
        ]
        
        last_error = None
        for model_id in preferred_models:
            try:
                logger.info("[DIARIZATION] Trying %s …", model_id)
                segments = _run_pyannote_with_timeout(model_id, token, wav_path, kwargs, PYANNOTE_TIMEOUT_SEC)
                segments = _merge_adjacent_segments(segments)
                logger.info("[DIARIZATION] pyannote (%s) → %d segments", model_id, len(segments))
                return segments if segments else None
            except TimeoutError:
                logger.error("[DIARIZATION] %s timed out", model_id)
                last_error = "timeout"
                break  # Don't retry other models if timeout - likely system issue
            except Exception as exc:
                logger.warning("[DIARIZATION] %s failed: %s", model_id, exc)
                last_error = exc
                continue  # Try next model
        
        if last_error == "timeout":
            return None  # Trigger fallback
        if last_error:
            logger.error("[DIARIZATION] All pyannote models failed: %s", last_error)
        return None

    except Exception as exc:
        logger.error("[DIARIZATION] pyannote wrapper failed: %s", exc, exc_info=True)
        return None


def _diarize_resemblyzer(wav_path: str, n_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore
        from sklearn.cluster import SpectralClustering  # type: ignore
    except ImportError:
        logger.warning("[DIARIZATION] resemblyzer/sklearn not installed")
        return None

    try:
        logger.info("[DIARIZATION] Starting resemblyzer fallback...")
        start = time.time()
        wav = preprocess_wav(wav_path)
        encoder = VoiceEncoder()
        sr = 16_000
        win = int(1.5 * sr)
        hop = int(0.75 * sr)

        total_chunks = max(1, (len(wav) - win) // hop + 1)
        embeds: list = []
        timestamps: list = []

        for i, start_idx in enumerate(range(0, len(wav) - win + 1, hop)):
            chunk = wav[start_idx : start_idx + win]
            embeds.append(encoder.embed_utterance(chunk))
            timestamps.append((start_idx / sr, (start_idx + win) / sr))
            
            if total_chunks > 10 and i > 0 and i % max(1, total_chunks // 10) == 0:
                logger.info("[DIARIZATION] resemblyzer progress: %d%% (%d/%d chunks)", 
                           int(100 * i / total_chunks), i, total_chunks)

        if not embeds:
            return None

        X = np.vstack(embeds)
        duration_sec = len(wav) / sr

        if n_speakers is None and (duration_sec < 120 or len(X) <= 4):
            segment = {"start": 0.0, "end": float(duration_sec), "speaker": "SPEAKER_00"}
            logger.info("[DIARIZATION] resemblyzer → 1 segment (single-speaker guard)")
            return [segment]

        if len(X) == 1:
            segment = {"start": float(timestamps[0][0]), "end": float(timestamps[0][1]), "speaker": "SPEAKER_00"}
            return [segment]

        if n_speakers and n_speakers > 0:
            k = min(n_speakers, len(X))
        else:
            k = min(5, max(1, int(len(X) ** 0.5)))
            k = min(k, len(X))

        if k <= 1:
            segments = [
                {"start": float(st), "end": float(en), "speaker": "SPEAKER_00"}
                for st, en in timestamps
            ]
            return _merge_adjacent_segments(segments)

        logger.info("[DIARIZATION] resemblyzer clustering %d speakers from %d chunks...", k, len(X))
        
        clustering = SpectralClustering(
            n_clusters=k,
            affinity="rbf",
            random_state=42,
            n_init=10,
        )
        labels = clustering.fit_predict(X)

        segments = [
            {"start": float(st), "end": float(en), "speaker": f"SPEAKER_{int(lab):02d}"}
            for (st, en), lab in zip(timestamps, labels)
        ]
        segments = _merge_adjacent_segments(segments)
        elapsed = time.time() - start
        logger.info("[DIARIZATION] resemblyzer → %d segments (%d speakers) in %.1fs", 
                   len(segments), k, elapsed)
        return segments

    except Exception as exc:
        logger.error("[DIARIZATION] resemblyzer failed: %s", exc, exc_info=True)
        return None


_NAME_PATTERNS = [
    r"\b(?:i\'m|i am|my name is|this is|name\'s)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    r"\b(?:i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
]


def extract_explicit_names_from_transcript(transcript: str) -> list[str]:
    names: list[str] = []
    if not transcript:
        return names
    for pattern in _NAME_PATTERNS:
        for m in re.finditer(pattern, transcript, flags=re.IGNORECASE):
            name = m.group(1).strip()
            if name and name not in names:
                names.append(name)
    return names


def infer_speaker_alias_map_from_transcript(transcript: str) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    if not transcript:
        return alias_map
    for raw_line in transcript.splitlines():
        m = re.match(r"^\s*((?:SPEAKER_\d+)|(?:Speaker\s+\d+)|(?:[A-Z]))\s*:\s*(.+)$", raw_line)
        if not m:
            continue
        speaker_label = m.group(1).strip()
        body = m.group(2).strip()
        for pattern in _NAME_PATTERNS:
            found = re.search(pattern, body, flags=re.IGNORECASE)
            if found:
                candidate = found.group(1).strip()
                if candidate and speaker_label not in alias_map:
                    alias_map[speaker_label] = candidate
                break
    return alias_map


def apply_speaker_aliases(
    segments: List[Dict[str, Any]],
    alias_map: dict[str, str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seg in segments:
        speaker = str(seg.get("speaker", "")).strip()
        alias = alias_map.get(speaker)
        item = dict(seg)
        item["speaker_label"] = speaker
        item["speaker_name"] = alias
        item["speaker_display"] = alias or speaker
        out.append(item)
    return out


def diarize_audio(wav_path: str, n_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    """Main entry point: try pyannote first, fallback to resemblyzer."""
    segments = _diarize_pyannote(wav_path, n_speakers)
    if segments is not None:
        return segments
    logger.info("[DIARIZATION] Falling back to resemblyzer...")
    return _diarize_resemblyzer(wav_path, n_speakers)