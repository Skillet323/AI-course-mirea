from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """Полный обзор датасета по колонкам."""
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []
    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)
        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )
        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None
        if is_numeric and non_null > 0:
            # явное приведение, чтобы не было numpy-типов
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())
        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )
    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица пропусков по колонкам: count/share."""
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])
    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        ).sort_values("missing_share", ascending=False)
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Корреляция Пирсона для числовых колонок."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame, max_columns: int = 5, top_k: int = 5
) -> Dict[str, pd.DataFrame]:
    """Для категориальных/строковых колонок считает top-k значений."""
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []
    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)
    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table
    return result


def compute_quality_flags(
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    min_missing_share: float = 0.1,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Эвристики качества данных:
    - слишком мало строк;
    - слишком много колонок;
    - максимальная доля пропусков;
    - константные колонки;
    - категориальные с высокой кардинальностью;
    - проблемные колонки по порогу пропусков;
    - подозрительные дубликаты id-полей;
    - много нулей в числовых колонках (если передан df).
    """
    flags: Dict[str, Any] = {}

    # базовые флаги
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5

    n_rows = summary.n_rows
    n_cols = summary.n_cols

    # 1) константные колонки
    constant_columns = [col.name for col in summary.columns if col.unique <= 1 and col.non_null > 0]
    flags["has_constant_columns"] = len(constant_columns) > 0
    flags["constant_columns"] = constant_columns

    # 2) высококардинальные категориальные признаки
    HIGH_CARDINALITY_THRESHOLD = max(50, int(max(10, n_rows * 0.1)))
    high_cardinality_columns = [
        col.name
        for col in summary.columns
        if (not col.is_numeric) and (col.unique > HIGH_CARDINALITY_THRESHOLD)
    ]
    flags["has_high_cardinality_categoricals"] = len(high_cardinality_columns) > 0
    flags["high_cardinality_columns"] = high_cardinality_columns
    flags["high_cardinality_threshold"] = HIGH_CARDINALITY_THRESHOLD

    # 3) проблемные колонки по пропускам (по порогу)
    problematic_missing_cols = []
    if not missing_df.empty:
        problematic_missing_cols = missing_df[missing_df["missing_share"] > min_missing_share].index.tolist()
    flags["problematic_missing_cols"] = problematic_missing_cols
    flags["problematic_missing_count"] = len(problematic_missing_cols)
    flags["min_missing_share"] = min_missing_share

    # 4) подозрительные id-колонки с дублями (ищем типичные имена *_id, id, user_id)
    suspicious_id_columns = []
    id_like_names = set()
    for col in summary.columns:
        lname = col.name.lower()
        if lname == "id" or lname.endswith("_id") or lname.startswith("id_") or lname in {"user_id", "uid"}:
            id_like_names.add(col.name)
    # если такие колонки есть — проверим уникальность по summary
    for name in id_like_names:
        col = next((c for c in summary.columns if c.name == name), None)
        if col is None:
            continue
        if col.non_null > 0 and col.unique < col.non_null:
            suspicious_id_columns.append(name)
    flags["has_suspicious_id_duplicates"] = len(suspicious_id_columns) > 0
    flags["suspicious_id_columns"] = suspicious_id_columns

    # 5) много нулей в числовых колонках (требует df для точного расчёта)
    zero_value_columns = []
    ZERO_VALUE_THRESHOLD = 0.5  # если >50% значений == 0 => тревожно
    if df is not None and not df.empty:
        numeric_cols = df.select_dtypes(include="number").columns
        for name in numeric_cols:
            s = df[name].dropna()
            if s.empty:
                continue
            zero_share = float((s == 0).sum() / len(s))
            if zero_share >= ZERO_VALUE_THRESHOLD:
                zero_value_columns.append({"column": name, "zero_share": zero_share})
    flags["has_many_zero_values"] = len(zero_value_columns) > 0
    flags["zero_value_columns"] = zero_value_columns
    flags["zero_value_threshold"] = ZERO_VALUE_THRESHOLD

    # --- расчёт интегрального показателя качества (score 0..1)
    score = 1.0
    # большая максимальная доля пропусков — сильный штраф
    score -= max_missing_share  # 0..1

    if flags["too_few_rows"]:
        score -= 0.15
    if flags["too_many_columns"]:
        score -= 0.05
    if flags["has_constant_columns"]:
        score -= 0.08
    if flags["has_high_cardinality_categoricals"]:
        score -= 0.10
    if flags["problematic_missing_count"] > 0:
        penalty = min(0.25, flags["problematic_missing_count"] * 0.04)
        score -= penalty
    if flags["has_suspicious_id_duplicates"]:
        score -= 0.08
    if flags["has_many_zero_values"]:
        # суммарный штраф, но не слишком большой
        score -= min(0.12, 0.06 * len(zero_value_columns))

    # clamp
    score = max(0.0, min(1.0, score))
    flags["quality_score"] = score

    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """Превращает DatasetSummary в табличку для более удобного вывода."""
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)
