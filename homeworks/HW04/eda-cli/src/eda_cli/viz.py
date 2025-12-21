from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def _ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_histograms_per_column(
    df: pd.DataFrame,
    out_dir: PathLike,
    max_columns: int = 6,
    bins: int = 20,
) -> List[Path]:
    """
    Для числовых колонок строит по отдельной гистограмме.
    Возвращает список путей к PNG.
    """
    out_dir = _ensure_dir(out_dir)
    numeric_df = df.select_dtypes(include="number")

    paths: List[Path] = []
    for i, name in enumerate(numeric_df.columns[:max_columns]):
        s = numeric_df[name].dropna()
        if s.empty:
            continue

        fig, ax = plt.subplots()
        ax.hist(s.values, bins=bins)
        ax.set_title(f"Histogram of {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
        fig.tight_layout()

        out_path = out_dir / f"hist_{i+1}_{name}.png"
        fig.savefig(out_path)
        plt.close(fig)

        paths.append(out_path)

    return paths


def plot_missing_matrix(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Простая визуализация пропусков: где True=пропуск, False=значение.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        # Рисуем пустой график
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Empty dataset", ha="center", va="center")
        ax.axis("off")
    else:
        mask = df.isna().values
        fig, ax = plt.subplots(figsize=(min(12, df.shape[1] * 0.4), 4))
        ax.imshow(mask, aspect="auto", interpolation="none")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        ax.set_title("Missing values matrix")
        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels(df.columns, rotation=90, fontsize=8)
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Тепловая карта корреляции числовых признаков.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough numeric columns for correlation", ha="center", va="center")
        ax.axis("off")
    else:
        corr = numeric_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(min(10, corr.shape[1]), min(8, corr.shape[0])))
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(corr.shape[1]))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(corr.shape[0]))
        ax.set_yticklabels(corr.index, fontsize=8)
        ax.set_title("Correlation heatmap")
        fig.colorbar(im, ax=ax, label="Pearson r")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def plot_categorical_distribution(df, column, output_path, top_k=10):
    """
    Создает столбчатую диаграмму распределения категорий для заданной колонки.
    
    Args:
        df: DataFrame с данными
        column: Имя категориальной колонки
        output_path: Путь для сохранения графика
        top_k: Количество топ-категорий для отображения
    
    Returns:
        Path к сохраненному изображению или None, если колонка не существует
    """
    if column not in df.columns:
        return None
        
    # Проверяем, что колонка категориальная или строковая
    s = df[column]
    if not (pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)):
        return None
        
    # Получаем топ-k категорий
    value_counts = s.value_counts().nlargest(top_k)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(value_counts.index.astype(str), value_counts.values)
    plt.title(f'Распределение значений в колонке "{column}"')
    plt.xlabel(column)
    plt.ylabel('Количество')
    plt.xticks(rotation=45, ha='right')
    
    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

def save_top_categories_tables(
    top_cats: Dict[str, pd.DataFrame],
    out_dir: PathLike,
) -> List[Path]:
    """
    Сохраняет top-k категорий по колонкам в отдельные CSV.
    """
    out_dir = _ensure_dir(out_dir)
    paths: List[Path] = []
    for name, table in top_cats.items():
        out_path = out_dir / f"top_values_{name}.csv"
        table.to_csv(out_path, index=False)
        paths.append(out_path)
    return paths
