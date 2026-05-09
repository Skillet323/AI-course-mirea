from src.services.evaluation import evaluate_tasks


def test_evaluate_tasks_perfect_match():
    pred = [
        {
            "description": "Prepare the final report",
            "assignee_hint": "Alice",
            "deadline_hint": "Friday",
        }
    ]
    gold = [
        {
            "description": "Prepare the final report",
            "assignee_hint": "Alice",
            "deadline_hint": "Friday",
        }
    ]

    metrics = evaluate_tasks(pred, gold)

    assert metrics["task_set_f1"] == 1.0
    assert metrics["task_set_precision"] == 1.0
    assert metrics["task_set_recall"] == 1.0
    assert metrics["assignee_accuracy"] == 1.0
    assert metrics["deadline_accuracy"] == 1.0
    assert metrics["matched_tasks"] == 1


def test_evaluate_tasks_partial_match():
    pred = [
        {
            "description": "Prepare the report",
            "assignee_hint": "Alice",
            "deadline_hint": "Friday",
        }
    ]
    gold = [
        {
            "description": "Prepare the final report",
            "assignee_hint": "Alice",
            "deadline_hint": "Friday",
        }
    ]

    metrics = evaluate_tasks(pred, gold)

    assert metrics["task_set_precision"] >= 0.0
    assert metrics["task_set_recall"] >= 0.0
    assert metrics["predicted_tasks"] == 1
    assert metrics["gold_tasks"] == 1
