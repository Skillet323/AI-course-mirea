def test_core_imports():
    from src.services.task_extraction import extract_tasks_rule_based
    from src.services.diarization import diarize_audio
    from src.services.transcription import _normalize_speaker_label

    assert callable(extract_tasks_rule_based)
    assert callable(diarize_audio)
    assert _normalize_speaker_label("SPEAKER_1") == "SPEAKER_01"
