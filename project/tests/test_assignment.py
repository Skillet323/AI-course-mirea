from src.models import Participant
from src.services.assignment_engine import assign_task_to_participant


def test_assign_by_assignee_hint():
    participants = [
        Participant(name="Alice Smith", email="alice@example.com", role="Designer"),
        Participant(name="Bob Jones", email="bob@example.com", role="Manager"),
    ]
    task = {
        "description": "Prepare the final design brief",
        "assignee_hint": "Alice Smith",
    }

    assignee, source, confidence = assign_task_to_participant(task, None, participants, 0)

    assert assignee == "Alice Smith"
    assert source == "assignee_hint"
    assert confidence >= 0.9


def test_assign_by_speaker_alias():
    participants = [
        Participant(name="Alice Smith", email="alice@example.com", role="Designer"),
        Participant(name="Bob Jones", email="bob@example.com", role="Manager"),
    ]
    meeting_info = {"speaker_aliases": {"SPEAKER_01": "Bob Jones"}}
    task = {
        "description": "Review the project timeline",
        "speaker_hint": "SPEAKER_01",
    }

    assignee, source, confidence = assign_task_to_participant(task, meeting_info, participants, 0)

    assert assignee == "Bob Jones"
    assert source == "speaker_alias"
    assert confidence >= 0.9


def test_unassigned_when_no_participants():
    task = {"description": "Prepare the summary report"}
    assignee, source, confidence = assign_task_to_participant(task, None, [], 0)

    assert assignee is None
    assert source == "unassigned"
    assert confidence == 0.0
