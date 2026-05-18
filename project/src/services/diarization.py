"""
Speaker diarization service — three-level fallback chain.

Level 1 — pyannote/speaker-diarization-3.1
    Best quality. Requires HF_TOKEN. Subprocess-isolated (avoids CUDA issues).
    Supports unlimited audio length via automatic chunking.

Level 2 — Resemblyzer + ECAPA-style embeddings (NEW intermediate)
    Installed-only path. Uses speechbrain pretrained ECAPA-TDNN for richer
    speaker embeddings when available, otherwise falls back to resemblyzer's
    own encoder. Silhouette-optimal AgglomerativeClustering (Ward linkage).

Level 3 — Resemblyzer voice encoder (original fallback)
    Always available when resemblyzer is installed.
    Silhouette-optimal k, AgglomerativeClustering.
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

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Pyannote config — NO hard duration limit; chunking handles any length
# Raised timeout: pyannote on GPU needs ~2-4x real-time for medium models.
# For 1140s audio processed in one shot: 300s * 2 chunks = 600s max.
PYANNOTE_TIMEOUT_PER_CHUNK_SEC = 300
PYANNOTE_CHUNK_SEC = 600           # 10-min chunks
PYANNOTE_CHUNK_OVERLAP_SEC = 30

# Resemblyzer window config
RESEMBLYZER_WIN_SEC = 1.5
RESEMBLYZER_HOP_SEC = 0.75
RESEMBLYZER_MAX_SPEAKERS = 8
RESEMBLYZER_MIN_SPEAKERS = 2


# ─────────────────────────────── helpers ──────────────────────────────────────

def _merge_adjacent_segments(segments: List[Dict[str, Any]], max_gap: float = 0.25) -> List[Dict[str, Any]]:
    """Merge same-speaker segments separated by a tiny gap."""
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and float(seg["start"]) - float(prev["end"]) <= max_gap:
            prev["end"] = max(float(prev["end"]), float(seg["end"]))
        else:
            merged.append(seg.copy())
    return merged


def _audio_info(wav_path: str) -> tuple[float, float]:
    """Return (duration_sec, size_mb)."""
    try:
        import soundfile as sf
        duration = float(sf.info(wav_path).duration or 0.0)
    except Exception:
        duration = 0.0
    size_mb = os.path.getsize(wav_path) / (1024 * 1024) if os.path.exists(wav_path) else 0.0
    return duration, size_mb


def _estimate_n_speakers_silhouette(X: np.ndarray) -> int:
    """Silhouette-score search over [MIN, MAX] speakers."""
    n = len(X)
    max_k = min(RESEMBLYZER_MAX_SPEAKERS, n - 1)
    if max_k < RESEMBLYZER_MIN_SPEAKERS:
        return 1
    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore
        from sklearn.metrics import silhouette_score         # type: ignore

        best_k, best_score = RESEMBLYZER_MIN_SPEAKERS, -1.0
        for k in range(RESEMBLYZER_MIN_SPEAKERS, max_k + 1):
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels, sample_size=min(1000, n), random_state=42)
            logger.debug("[DIARIZATION] silhouette k=%d → %.4f", k, score)
            if score > best_score:
                best_score, best_k = score, k

        logger.info("[DIARIZATION] Silhouette-optimal k=%d (score=%.4f)", best_k, best_score)
        return best_k
    except Exception as exc:
        logger.warning("[DIARIZATION] silhouette search failed (%s), using heuristic k", exc)
        return min(5, max(2, int(n ** 0.4)))


def _cluster(X: np.ndarray, k: int) -> np.ndarray:
    """AgglomerativeClustering (Ward); SpectralClustering as last resort."""
    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    except Exception:
        from sklearn.cluster import SpectralClustering  # type: ignore
        return SpectralClustering(n_clusters=k, affinity="rbf", random_state=42, n_init=10).fit_predict(X)


# ─────────────────────────── Level 1: pyannote ────────────────────────────────

def _try_annotation(obj) -> Optional[list]:
    """Try to extract [(start, end, speaker)] from a pyannote Annotation-like object."""
    if obj is None:
        return None
    if callable(getattr(obj, 'itertracks', None)):
        try:
            return [
                (float(t.start), float(t.end), str(s))
                for t, _, s in obj.itertracks(yield_label=True)
            ]
        except Exception:
            pass
    return None


def _iter_result(result) -> list:
    """
    Robustly extract [(start, end, speaker)] from ANY pyannote pipeline output.

    Handles:
    - Standard pyannote Annotation (.itertracks)             → pyannote ≤3
    - DiarizeOutput(diarization=Annotation, ...)             → pyannote 3.x
    - Any other wrapper: scans all public non-callable attrs
    - .segments list fallback
    - Iterable-of-dicts fallback
    """
    # 1. Direct Annotation
    segs = _try_annotation(result)
    if segs is not None:
        return segs

    # 2. Known wrapper attribute names
    for attr_name in ('diarization', 'annotation', 'speaker_diarization',
                      'output', 'result', 'speakers'):
        segs = _try_annotation(getattr(result, attr_name, None))
        if segs is not None:
            return segs

    # 3. Full attribute scan (handles any future renaming in pyannote)
    for attr_name in dir(result):
        if attr_name.startswith('_'):
            continue
        try:
            attr = getattr(result, attr_name)
        except Exception:
            continue
        if attr is result or callable(attr):
            continue
        segs = _try_annotation(attr)
        if segs is not None:
            logger.debug("[DIARIZATION] Found Annotation via attr %r", attr_name)
            return segs

    # 4. .segments list
    if hasattr(result, 'segments'):
        out = []
        for seg in result.segments:
            start = getattr(seg, 'start', None) or getattr(seg, 'onset', None)
            end   = getattr(seg, 'end',   None) or getattr(seg, 'offset', None)
            spk   = getattr(seg, 'speaker', None) or getattr(seg, 'label', 'SPEAKER_00')
            if start is not None and end is not None:
                out.append((float(start), float(end), str(spk)))
        if out:
            return out

    # 5. Iterable-of-dicts
    try:
        out = []
        for item in result:
            if isinstance(item, dict):
                out.append((
                    float(item.get('start', 0)),
                    float(item.get('end',   0)),
                    str(item.get('speaker', 'SPEAKER_00')),
                ))
        if out:
            return out
    except TypeError:
        pass

    raise ValueError(
        f"Cannot parse diarization result of type {type(result)}. "
        f"Public attrs: {[a for a in dir(result) if not a.startswith('_')]}"
    )


def _pyannote_subprocess(model_id: str, token: str, wav_path: str,
                         kwargs: dict, result_queue: mp.Queue):
    """
    Run pyannote inside a clean subprocess.
    Automatically chunks audio longer than PYANNOTE_CHUNK_SEC.
    """
    import traceback
    try:
        from pyannote.audio import Pipeline  # type: ignore
        try:
            pipeline = Pipeline.from_pretrained(model_id, token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline.to(torch.device(device))

        import soundfile as sf  # type: ignore
        info = sf.info(wav_path)
        duration, sr = float(info.duration), int(info.samplerate)
        all_segments: List[Dict[str, Any]] = []

        if duration <= PYANNOTE_CHUNK_SEC:
            # Short enough — process in one shot
            data, _ = sf.read(wav_path, dtype="float32")
            wav = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data).permute(1, 0)
            result = pipeline({"waveform": wav, "sample_rate": sr}, **kwargs)
            for start, end, spk in _iter_result(result):
                all_segments.append({"start": start, "end": end, "speaker": spk})
        else:
            # Chunked processing for long audio
            n_chunks = int(duration / PYANNOTE_CHUNK_SEC) + 1
            logger.info("[DIARIZATION-SUB] %.0fs audio → %d chunks of %ds", duration, n_chunks, PYANNOTE_CHUNK_SEC)
            global_counter = 0
            chunk_start = 0.0
            chunk_idx = 0
            global_spk_map: dict[tuple, str] = {}

            while chunk_start < duration:
                chunk_end = min(chunk_start + PYANNOTE_CHUNK_SEC, duration)
                read_start = max(0.0, chunk_start - PYANNOTE_CHUNK_OVERLAP_SEC)
                read_end = min(chunk_end + PYANNOTE_CHUNK_OVERLAP_SEC, duration)

                data, _ = sf.read(wav_path,
                                  start=int(read_start * sr),
                                  frames=int((read_end - read_start) * sr),
                                  dtype="float32")
                wav = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data).permute(1, 0)
                result = pipeline({"waveform": wav, "sample_rate": sr}, **kwargs)

                for abs_start, abs_end, spk in _iter_result(result):
                    abs_start += read_start
                    abs_end += read_start
                    seg_s = max(abs_start, chunk_start)
                    seg_e = min(abs_end, chunk_end)
                    if seg_e <= seg_s:
                        continue
                    key = (chunk_idx, spk)
                    if key not in global_spk_map:
                        global_spk_map[key] = f"SPEAKER_{global_counter:02d}"
                        global_counter += 1
                    all_segments.append({"start": seg_s, "end": seg_e,
                                         "speaker": global_spk_map[key]})

                chunk_start = chunk_end
                chunk_idx += 1
                logger.info("[DIARIZATION-SUB] chunk %d/%d done, %d segs so far",
                            chunk_idx, n_chunks, len(all_segments))

            all_segments.sort(key=lambda s: s["start"])

        result_queue.put({"ok": True, "segments": all_segments})
    except Exception as exc:
        result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}\n{''.join(__import__('traceback').format_tb(exc.__traceback__))}"})


def _run_pyannote(model_id: str, token: str, wav_path: str,
                  kwargs: dict, duration_sec: float) -> List[Dict[str, Any]]:
    """Spawn subprocess; timeout scales with number of chunks."""
    n_chunks = max(1, int(duration_sec / PYANNOTE_CHUNK_SEC) + 1)
    timeout = PYANNOTE_TIMEOUT_PER_CHUNK_SEC * n_chunks

    q: mp.Queue = mp.Queue()
    proc = mp.Process(target=_pyannote_subprocess, args=(model_id, token, wav_path, kwargs, q))
    logger.info("[DIARIZATION] pyannote subprocess starting (model=%s, timeout=%ds)", model_id, timeout)
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate(); proc.join(5)
        if proc.is_alive(): proc.kill()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        raise TimeoutError(f"pyannote exceeded {timeout}s")

    if q.empty():
        raise RuntimeError("pyannote subprocess produced no result")
    outcome = q.get()
    if not outcome["ok"]:
        raise RuntimeError(outcome["error"])
    return outcome["segments"]


def _diarize_pyannote(wav_path: str, n_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    token: str = settings.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not token:
        logger.warning("[DIARIZATION] HF_TOKEN not set — skipping pyannote")
        return None

    duration_sec, size_mb = _audio_info(wav_path)
    logger.info("[DIARIZATION] pyannote attempt: duration=%.0fs, size=%.1fMB", duration_sec, size_mb)

    kwargs: Dict[str, Any] = {}
    if n_speakers is not None:
        kwargs["num_speakers"] = n_speakers

    models = [
        "pyannote/speaker-diarization-3.1",
        "pyannote-community/speaker-diarization-community-1",
        "pyannote/speaker-diarization-community-1",
    ]

    last_error = None
    for model_id in models:
        try:
            logger.info("[DIARIZATION] Trying %s …", model_id)
            segs = _run_pyannote(model_id, token, wav_path, kwargs, duration_sec)
            segs = _merge_adjacent_segments(segs)
            logger.info("[DIARIZATION] pyannote (%s) → %d segments", model_id, len(segs))
            return segs or None
        except TimeoutError:
            logger.error("[DIARIZATION] %s timed out — skipping remaining pyannote models", model_id)
            break
        except Exception as exc:
            err_str = str(exc)
            if "GatedRepoError" in err_str or "403" in err_str or "gated" in err_str.lower():
                logger.error(
                    "[DIARIZATION] %s is a gated model — access denied (403).\n"
                    "  → Accept conditions at https://hf.co/pyannote/segmentation-3.0\n"
                    "  → Accept conditions at https://hf.co/pyannote/speaker-diarization-3.1\n"
                    "  → Make sure HF_TOKEN in .env has 'read' permission",
                    model_id,
                )
            else:
                logger.warning("[DIARIZATION] %s failed: %s", model_id, exc)
            last_error = exc

    if last_error:
        logger.error("[DIARIZATION] All pyannote models failed, last error: %s", last_error)
    return None


# ──────── Level 2: ECAPA-TDNN embeddings (intermediate, richer than resemblyzer) ─────

def _embed_ecapa(wav_chunks: list, device: str) -> Optional[np.ndarray]:
    """
    Embed audio chunks with SpeechBrain's ECAPA-TDNN speaker encoder.
    Returns an (N, D) numpy array or None if not available.
    """
    try:
        from speechbrain.pretrained import EncoderClassifier  # type: ignore
        import torchaudio  # type: ignore

        logger.info("[DIARIZATION] Loading ECAPA-TDNN speaker encoder...")
        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        embeds = []
        for chunk in wav_chunks:
            t = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
            with torch.no_grad():
                emb = encoder.encode_batch(t)
            embeds.append(emb.squeeze().cpu().numpy())
        return np.vstack(embeds) if embeds else None
    except Exception as exc:
        logger.info("[DIARIZATION] ECAPA-TDNN not available (%s), using resemblyzer encoder", exc)
        return None


def _diarize_ecapa(wav_path: str, n_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Intermediate diarization using SpeechBrain ECAPA-TDNN embeddings.
    Falls through (returns None) if speechbrain is not installed.
    """
    try:
        from resemblyzer import preprocess_wav  # type: ignore
    except ImportError:
        return None

    try:
        logger.info("[DIARIZATION] Starting ECAPA-TDNN intermediate diarizer...")
        t0 = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        wav = preprocess_wav(wav_path)
        sr = 16_000
        win = int(RESEMBLYZER_WIN_SEC * sr)
        hop = int(RESEMBLYZER_HOP_SEC * sr)
        duration_sec = len(wav) / sr

        chunks, timestamps = [], []
        for si in range(0, len(wav) - win + 1, hop):
            chunks.append(wav[si: si + win])
            timestamps.append((si / sr, (si + win) / sr))

        if not chunks:
            return None

        # Try ECAPA embeddings first
        X = _embed_ecapa(chunks, device)

        if X is None:
            return None  # No speechbrain; caller will try resemblyzer

        if len(X) == 1:
            return [{"start": float(timestamps[0][0]), "end": float(timestamps[0][1]), "speaker": "SPEAKER_00"}]

        # Trivial single-speaker guard
        if duration_sec < 60 and len(X) <= 4 and n_speakers is None:
            return _merge_adjacent_segments([
                {"start": float(st), "end": float(en), "speaker": "SPEAKER_00"}
                for st, en in timestamps
            ])

        k = n_speakers if (n_speakers and n_speakers > 0) else _estimate_n_speakers_silhouette(X)
        k = max(1, min(k, len(X) - 1))

        if k <= 1:
            return _merge_adjacent_segments([
                {"start": float(st), "end": float(en), "speaker": "SPEAKER_00"}
                for st, en in timestamps
            ])

        logger.info("[DIARIZATION] ECAPA-TDNN clustering k=%d on %d chunks...", k, len(X))
        labels = _cluster(X, k)

        segments = [
            {"start": float(st), "end": float(en), "speaker": f"SPEAKER_{int(lab):02d}"}
            for (st, en), lab in zip(timestamps, labels)
        ]
        segments = _merge_adjacent_segments(segments)
        logger.info("[DIARIZATION] ECAPA-TDNN → %d segments (%d speakers) in %.1fs",
                    len(segments), k, time.time() - t0)
        return segments

    except Exception as exc:
        logger.error("[DIARIZATION] ECAPA-TDNN diarizer failed: %s", exc, exc_info=True)
        return None


# ──────────────── Level 3: resemblyzer (original fallback) ────────────────────

def _diarize_resemblyzer(wav_path: str, n_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore
    except ImportError:
        logger.warning("[DIARIZATION] resemblyzer not installed")
        return None

    try:
        logger.info("[DIARIZATION] Starting resemblyzer fallback...")
        t0 = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        wav = preprocess_wav(wav_path)
        encoder = VoiceEncoder(device)
        sr = 16_000
        win = int(RESEMBLYZER_WIN_SEC * sr)
        hop = int(RESEMBLYZER_HOP_SEC * sr)
        duration_sec = len(wav) / sr

        total_chunks = max(1, (len(wav) - win) // hop + 1)
        embeds, timestamps = [], []
        for i, si in enumerate(range(0, len(wav) - win + 1, hop)):
            embeds.append(encoder.embed_utterance(wav[si: si + win]))
            timestamps.append((si / sr, (si + win) / sr))
            if total_chunks > 10 and i > 0 and i % max(1, total_chunks // 10) == 0:
                logger.info("[DIARIZATION] resemblyzer progress: %d%% (%d/%d chunks)",
                            int(100 * i / total_chunks), i, total_chunks)

        if not embeds:
            return None

        X = np.vstack(embeds)

        if duration_sec < 60 and len(X) <= 4 and n_speakers is None:
            return _merge_adjacent_segments([
                {"start": float(st), "end": float(en), "speaker": "SPEAKER_00"}
                for st, en in timestamps
            ])

        if len(X) == 1:
            return [{"start": float(timestamps[0][0]), "end": float(timestamps[0][1]), "speaker": "SPEAKER_00"}]

        if n_speakers and n_speakers > 0:
            k = min(n_speakers, len(X) - 1)
        else:
            k = _estimate_n_speakers_silhouette(X)

        k = max(1, min(k, len(X) - 1))

        if k <= 1:
            return _merge_adjacent_segments([
                {"start": float(st), "end": float(en), "speaker": "SPEAKER_00"}
                for st, en in timestamps
            ])

        logger.info("[DIARIZATION] resemblyzer clustering k=%d on %d chunks...", k, len(X))
        labels = _cluster(X, k)

        segments = [
            {"start": float(st), "end": float(en), "speaker": f"SPEAKER_{int(lab):02d}"}
            for (st, en), lab in zip(timestamps, labels)
        ]
        segments = _merge_adjacent_segments(segments)
        logger.info("[DIARIZATION] resemblyzer → %d segments (%d speakers) in %.1fs",
                    len(segments), k, time.time() - t0)
        return segments

    except Exception as exc:
        logger.error("[DIARIZATION] resemblyzer failed: %s", exc, exc_info=True)
        return None


# ───────────────────── name-inference helpers ──────────────────────────────────

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


def apply_speaker_aliases(segments: List[Dict[str, Any]], alias_map: dict[str, str]) -> List[Dict[str, Any]]:
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


# ───────────────────────────── public entry ───────────────────────────────────

def diarize_audio(wav_path: str, n_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Three-level fallback:
        Level 1 — pyannote (requires HF_TOKEN; any audio length via chunking)
        Level 2 — ECAPA-TDNN via SpeechBrain (if installed; richer embeddings)
        Level 3 — resemblyzer (always available; silhouette-optimal k)
    """
    # Level 1
    segs = _diarize_pyannote(wav_path, n_speakers)
    if segs is not None:
        return segs

    # Level 2
    logger.info("[DIARIZATION] Trying ECAPA-TDNN intermediate model...")
    segs = _diarize_ecapa(wav_path, n_speakers)
    if segs is not None:
        return segs

    # Level 3
    logger.info("[DIARIZATION] Falling back to resemblyzer...")
    return _diarize_resemblyzer(wav_path, n_speakers)
