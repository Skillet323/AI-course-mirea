# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Команды

### overview
Краткий обзор датасета: размеры, типы колонок, базовая статистика.

```bash
uv run eda-cli overview data/example.csv
```

### report
Генерация полного EDA-отчёта в формате Markdown с визуализациями.

```bash
uv run eda-cli report data/example.csv --out-dir reports_example
```

#### Дополнительные параметры команды report:

- `--max-hist-columns`: Максимальное количество числовых колонок для отображения гистограмм (по умолчанию: 6)
- `--top-k-categories`: Количество топ-значений для отображения в категориальных признаках (по умолчанию: 10)
- `--min-missing-share`: Порог доли пропусков, при превышении которого колонка считается проблемной (по умолчанию: 0.1 = 10%)
- `--title`: Заголовок отчёта (по умолчанию: "EDA-отчёт")

Пример использования с кастомными параметрами:

```bash
uv run eda-cli report data/example.csv \
  --out-dir reports_custom \
  --max-hist-columns 4 \
  --top-k-categories 5 \
  --min-missing-share 0.2 \
  --title "Анализ данных клиентов"
```

## Структура отчёта

Отчёт включает:
- Общую информацию о датасете
- Анализ качества данных с эвристиками
- Статистику по пропускам
- Корреляционную матрицу для числовых признаков
- Распределение категориальных признаков
- Гистограммы для числовых колонок


## Тесты

```bash
uv run pytest -q
```

## HTTP API (HW04)

Запускаем сервер:
uv run uvicorn eda_cli.api:app --reload --port 8000

Эндпоинты:
- GET /health — статус сервиса.
- POST /quality — упрощённый JSON-эндпоинт (для быстрых проверок).
- POST /quality-from-csv — принимает CSV (multipart/form-data), возвращает quality_score и flags.
- POST /quality-flags-from-csv — **новый** эндпоинт HW04: принимает CSV и возвращает полный набор эвристик качества (включая эвристики из HW03: `has_constant_columns`, `has_high_cardinality_categoricals`, `has_suspicious_id_duplicates`, `has_many_zero_values` и т.д.)

Пример:
curl -F "file=@data/example.csv" "http://127.0.0.1:8000/quality-flags-from-csv?min_missing_share=0.1"

