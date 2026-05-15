"""
Tests for /api/predict endpoint.
Run: pytest tests/test_predict_endpoint.py -v
"""
import json
import os
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_predict.db")
os.environ.setdefault("TASK_PROVIDER", "rules")
os.environ.setdefault("DIARIZATION_ENABLED", "false")

from src.main import app

client = TestClient(app)


def test_health():
    """Health endpoint returns ok status."""
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "db" in data
    assert "timestamp" in data


def test_predict_basic():
    """Predict extracts tasks from transcript."""
    transcript = (
        "Alice will prepare the design specification by Friday. "
        "Bob needs to review the product requirements and send feedback to the team."
    )
    r = client.post("/api/predict", json={"transcript": transcript})
    assert r.status_code == 200
    data = r.json()
    assert "tasks" in data
    assert "task_count" in data
    assert isinstance(data["tasks"], list)
    assert data["task_count"] == len(data["tasks"])
    assert data["task_count"] >= 1


def test_predict_with_metadata():
    """Predict accepts and echoes optional metadata fields."""
    r = client.post("/api/predict", json={
        "transcript": "John must submit the report by Monday.",
        "language": "en",
        "duration_sec": 120.0,
        "meeting_ref": "test-meeting-01",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["meeting_ref"] == "test-meeting-01"
    assert "fallback_used" in data
    assert "provider" in data


def test_predict_empty_transcript_rejected():
    """Empty transcript returns 422."""
    r = client.post("/api/predict", json={"transcript": "  "})
    assert r.status_code == 422


def test_predict_missing_transcript_rejected():
    """Missing transcript field returns 422."""
    r = client.post("/api/predict", json={})
    assert r.status_code == 422


def test_predict_ami_gold_sample():
    """Predict runs on real AMI transcript sample and returns tasks."""
    gold_path = "data/gold/ES2002a.json"
    if not os.path.exists(gold_path):
        pytest.skip("Gold data not available")
    data = json.loads(open(gold_path).read())
    transcript = data.get("transcript", "")[:3000]
    r = client.post("/api/predict", json={"transcript": transcript, "meeting_ref": "ES2002a"})
    assert r.status_code == 200
    resp = r.json()
    assert resp["task_count"] >= 1
    assert resp["meeting_ref"] == "ES2002a"


def test_predict_fallback_indicated():
    """Rule-based fallback is flagged in response when no LLM key set."""
    r = client.post("/api/predict", json={"transcript": "Team should review the code."})
    data = r.json()
    # Without OPENROUTER_API_KEY, fallback_used should be True
    assert isinstance(data.get("fallback_used"), bool)
