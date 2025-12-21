from __future__ import annotations

import pandas as pd
import pytest

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, min_missing_share=0.1)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_quality_flags_constant_columns():
    """Test detection of constant columns."""
    df = pd.DataFrame({
        'constant_col': [1, 1, 1, 1],
        'normal_col': [1, 2, 3, 4]
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    assert flags['has_constant_columns'] is True


def test_quality_flags_no_constant_columns():
    """Test when there are no constant columns."""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': ['a', 'b', 'c', 'd']
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    assert flags['has_constant_columns'] is False


def test_quality_flags_high_cardinality():
    """Test detection of high cardinality categorical columns."""
    # Создаем DataFrame с высококардинальной категориальной колонкой
    high_card_data = [f'cat_{i}' for i in range(60)]
    df = pd.DataFrame({
        'high_card_col': high_card_data,
        'normal_col': [1] * 60
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    assert flags['has_high_cardinality_categoricals'] is True


def test_quality_flags_normal_cardinality():
    """Test when categorical columns have normal cardinality."""
    normal_card_data = ['cat_a', 'cat_b', 'cat_c'] * 20
    df = pd.DataFrame({
        'normal_card_col': normal_card_data,
        'num_col': range(60)
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    assert flags['has_high_cardinality_categoricals'] is False


def test_quality_flags_problematic_missing():
    """Test detection of columns with high missing rate."""
    df = pd.DataFrame({
        'good_col': [1, 2, 3, 4, 5],
        'bad_col1': [None, None, 3, 4, 5],  # 40% пропусков
        'bad_col2': [None, None, None, 4, 5]  # 60% пропусков
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Порог 50%
    flags1 = compute_quality_flags(summary, missing_df, min_missing_share=0.5)
    assert len(flags1['problematic_missing_cols']) == 1
    assert 'bad_col2' in flags1['problematic_missing_cols']
    
    # Порог 30%
    flags2 = compute_quality_flags(summary, missing_df, min_missing_share=0.3)
    assert len(flags2['problematic_missing_cols']) == 2
    assert 'bad_col1' in flags2['problematic_missing_cols']
    assert 'bad_col2' in flags2['problematic_missing_cols']

def test_compute_quality_flags_new_heuristics():
    df = pd.DataFrame({
        "user_id": [1, 1, 2, 3, 3],  # дубликаты: unique < non_null
        "const": [5, 5, 5, 5, 5],    # константа
        "val": [0, 0, 0, 1, 2],      # 60% нулей
        "maybe": ["a", None, "b", "a", "c"],
    })

    summary = summarize_dataset(df)
    missing = missing_table(df)
    flags = compute_quality_flags(summary, missing, min_missing_share=0.1, df=df)

    assert flags["has_constant_columns"] is True
    assert "const" in flags.get("constant_columns", [])
    assert flags["has_suspicious_id_duplicates"] is True
    assert "user_id" in flags.get("suspicious_id_columns", [])
    assert flags["has_many_zero_values"] is True
    # zero_value_columns содержит словари с полем column
    zero_cols = [x["column"] for x in flags.get("zero_value_columns", [])]
    assert "val" in zero_cols
    assert "quality_score" in flags
    assert 0.0 <= flags["quality_score"] <= 1.0