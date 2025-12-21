# tests/test_api.py
import io

import pandas as pd
from fastapi.testclient import TestClient

from eda_cli.api import app

client = TestClient(app)

def make_csv_bytes(dataframe):
    buf = io.BytesIO()
    dataframe.to_csv(buf, index=False)
    buf.seek(0)
    return buf

def test_quality_flags_from_csv_returns_flags():
    df = pd.DataFrame({
        "user_id": [1, 1, 2, 3, 3],
        "const": [5, 5, 5, 5, 5],
        "val": [0, 0, 0, 1, 2],
        "maybe": ["a", None, "b", "a", "c"],
    })
    csv_file = make_csv_bytes(df)
    files = {"file": ("test.csv", csv_file, "text/csv")}
    resp = client.post("/quality-flags-from-csv?min_missing_share=0.1", files=files)
    assert resp.status_code == 200, resp.text
    json_resp = resp.json()
    assert "flags" in json_resp
    # Проверяем, что наши новые эвристики представлены
    flags = json_resp["flags"]
    assert "has_constant_columns" in flags
    assert "has_suspicious_id_duplicates" in flags
    assert "has_many_zero_values" in flags
    assert "quality_score" in json_resp
    assert "ok_for_model" in json_resp
