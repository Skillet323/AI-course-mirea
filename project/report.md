# Отчёт по итоговому проекту по курсу «Инженерия Искусственного Интеллекта»

> Рекомендуемый объём отчёта: 3-5 страниц в эквиваленте Markdown/печатного текста.  
> Отчёт должен позволить преподавателю понять задачу, данные, выбранные модели и результаты экспериментов.

---

## 1. Паспорт проекта

- **Название проекта:** `Meeting Secretary`
- **Автор:** `Боргачев Тимофей Максимович`
- **Группа:** `ИНБО-20-23`
- **Контакт:** `@Skillet323`
- **Ссылка на репозиторий:** `https://github.com/Skillet323/AI-course-mirea`

Кратко: проект представляет собой веб-сервис для автоматической обработки совещаний.  
Система распознаёт речь, выполняет diarization, извлекает поручения, сопоставляет их с участниками, сохраняет результаты и позволяет сравнивать конфигурации моделей по метрикам.

---

## 2. Постановка задачи и контекст

Проект решает задачу автоматической обработки аудиозаписей совещаний. Пользователь загружает запись встречи, после чего система:

- преобразует аудио в текст;
- выделяет поручения и action items;
- определяет ответственных;
- сохраняет результат в базе данных;
- показывает метрики и диагностические признаки обработки.

Потенциальный пользователь — сотрудник или аналитик, которому важно быстро получить краткий и структурированный итог совещания без ручного протоколирования.

### Формулировка задачи в терминах ML/ИИ

В проекте используются несколько связанных задач:

- **ASR**: преобразование аудио в текст (Whisper);
- **speaker diarization**: разделение реплик по спикерам (pyannote / resemblyzer);
- **information extraction**: извлечение задач из транскрипта (LLM + rule-based fallback);
- **entity/role matching**: сопоставление задач с участниками (assignment engine).

Ограничения:
- шумные записи и перекрывающиеся реплики;
- нестабильное качество аудио;
- неявная формулировка поручений в разговоре;
- возможные rate limits на внешних LLM API.

### Целевые метрики качества

Используются:
- **WER** и **CER** — качество распознавания речи;
- **Task Precision**, **Task Recall**, **Task F1** — качество извлечения задач;
- **Assignee Accuracy** и **Deadline Accuracy** — точность атрибутов задач;
- **latency** и вспомогательные диагностические признаки — для анализа поведения пайплайна.

---

## 3. Данные

В проекте используются данные AMI Meeting Corpus.

### Источник данных

- Публичный датасет **AMI Meeting Corpus**, серия ES (открытая лицензия).
- 13 встреч (`ES2002a–ES2014a`), продолжительность 5–23 минуты.
- Gold-аннотации: 37 задач итого (≈ 2.85 задачи на встречу), ~77% задач имеют `assignee_hint`, ~69% — `deadline_hint`.

### Структура данных

- `data/audio/` — аудиозаписи;
- `data/gold/` — эталонные аннотации (transcript + tasks) в JSON;
- `artifacts/` — результаты сравнения моделей и evaluation.

### Предобработка и EDA

Подробный EDA выполнен в `notebooks/01_eda.ipynb`.

На этапе подготовки данных выполнялись:
- нормализация аудио и приведение к единой частоте дискретизации;
- транскрибация через Whisper small;
- анализ качества распознавания (WER/CER по gold);
- сравнение output по gold-разметке;
- анализ распределения количества задач по встречам и лексического покрытия.

Наблюдения:
- качество записей сильно различается (WER от 0.30 до 0.75);
- gold-задачи сформулированы абстрактно, не как прямые цитаты из разговора — это затрудняет rule-based matching;
- среднее лексическое покрытие задачи транскриптом — ~55%, что объясняет сложность задачи;
- у части встреч поручения сформулированы неявно, что требует LLM для корректного извлечения.

---

## 4. Модели и подходы

### Базовые модели и подходы

В проекте использовались:
- **Whisper** (`medium`, CPU) — для ASR;
- **pyannote/speaker-diarization-3.1** (первичный) / **resemblyzer** (fallback) — для diarization;
- **rule-based экстрактор** — baseline для извлечения задач (40+ маркеров действий: `should`, `must`, `need to`, `design`, `prepare`, `review`, `will`, `going to` и др.);
- **OpenRouter LLM-модели** — улучшенный подход для извлечения задач.

### Улучшения и эксперименты

В ходе доработки были добавлены:
- recursive parser для ответа LLM (устойчивость к нестандартному JSON);
- фильтрация meta-output и нерелевантного контекста;
- conservative mode для коротких/шумных записей;
- проверка согласованности задачи с транскриптом (`_task_supported_by_transcript`);
- speaker aliases для замены `SPEAKER_XX` на имена;
- global greedy matching в evaluation (матрица сходств, сортировка по убыванию score);
- endpoint `POST /api/predict` для прямого ML-инференса по транскрипту.

### Нейросетевые модели

- **ASR:** Whisper small (82M параметров, мультиязычный).
- **Task extraction:** LLM через OpenRouter API — сравнивались 4 модели (см. раздел 5).
- **Diarization:** pyannote/speaker-diarization-3.1 (primary) / resemblyzer + SpectralClustering (fallback).

---

## 5. Экспериментальный протокол и результаты

### Экспериментальный протокол

Сравнение моделей выполнялось на одном и том же наборе из 13 встреч AMI.  
Для оценки использовались gold-аннотации (`data/gold/`).  
Результаты сохранялись в `artifacts/model_comparison.csv` и `artifacts/model_comparison.json`.  
Воспроизводимый анализ — в `notebooks/02_baselines.ipynb`.

### Сравнение моделей по метрикам

| Модель / конфигурация | Успешных встреч | Всего встреч | Task F1 | Precision | Recall | Assignee Acc. | Deadline Acc. | Комментарий |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Rule-based (baseline)** | 0 | 13 | 0.140 | 0.080 | 0.628 | — | — | Всегда работает, низкий результат|
| `inclusionai/ring-2.6-1t:free` | 7 | 13 | **0.257** | **0.244** | **0.286** | 0.300 | 0.200 | Лучший баланс F1 |
| `openai/gpt-oss-120b:free` | 3 | 13 | 0.222 | 0.167 | 0.333 | 0.000 | 0.000 | Работает, но нестабилен |
| `nvidia/nemotron-3-super-120b-a12b:free` | 4 | 13 | 0.028 | 0.017 | 0.083 | 0.000 | 0.000 | Низкая полезность |

### Сводные метрики ASR (Whisper small, 13 встреч)

| Метрика | Среднее значение |
|---|---:|
| WER | 0.514 |
| CER | 0.357 |
| Predicted Tasks (avg) | 6.23 |
| Gold Tasks (avg) | 2.85 |

### Интерпретация результатов

- **Rule-based baseline:** высокий recall (0.63) — находит реплики с маркерами действий, но крайне низкий precision (0.08) — выдаёт ~23 кандидата при 3 gold-задачах.
- **LLM (`inclusionai`):** лучший Task F1=0.257 при 7/13 успешных встречах. Хороший баланс precision/recall по сравнению с baseline.
- **Ключевая проблема LLM:** нестабильность — 6/13 встреч закончились `parse_failed` (модель вернула нестандартный формат). При этом включается автоматический rule-based fallback.
- **WER=0.51** объясняется шумом в записях AMI и разговорным стилем речи. Whisper `large-v3` даст ~0.30, но требует GPU.

### Выбор финальной модели

**Финальная модель: `inclusionai/ring-2.6-1t:free` через OpenRouter.**

Обоснование:
1. Наивысший Task F1 (0.257) при достаточном числе успешных запусков (7/13) — лучший баланс качество/надёжность.
2. Значительно лучший precision (0.244) по сравнению с baseline (0.080) — меньше ложных срабатываний.
3. Бесплатный tier OpenRouter, без rate limits.
4. При сбоях (parse_failed) автоматически включается rule-based fallback, что гарантирует наличие результата.

Rule-based baseline используется только как fallback — его высокий recall (0.63) компенсирует низкий precision при недоступности LLM.

---

## 6. Архитектура решения и сервис

Пайплайн проекта:

```
Аудио → Preprocessing → ASR (Whisper)
      → Diarization (pyannote / resemblyzer)
      → Speaker Transcript
      → Task Extraction (LLM → fallback: rule-based)
      → Assignment Engine
      → DB (SQLite) → Evaluation → React UI
```

### API и endpoints

| Endpoint | Метод | Описание |
|---|---|---|
| `/api/predict` | POST | **ML inference**: `transcript` → `tasks` (основной predict endpoint) |
| `/api/upload_meeting` | POST | Полный пайплайн: аудиофайл → задачи |
| `/api/health` | GET | Статус сервиса + проверка БД |
| `/api/meeting/{id}` | GET | Детали встречи |
| `/api/meeting/{id}/progress` | GET | Прогресс обработки |
| `/api/meeting/{id}/export` | GET | Экспорт в JSON/CSV |
| `/api/metrics` | GET | Последние метрики (latency, quality) |
| `/api/stats` | GET | Агрегированная статистика |
| `/api/evaluations` | GET | Результаты оценки по gold |

### Технологический стек

- **Backend:** FastAPI + SQLModel + SQLite/PostgreSQL
- **ASR:** `openai-whisper` (small, CPU/GPU)
- **Diarization:** `pyannote.audio` (primary) / `resemblyzer` (fallback)
- **Task extraction:** OpenRouter API + rule-based fallback
- **Frontend:** React + Vite
- **Контейнеризация:** Docker (multi-stage build: Node → Python)

---

## 7. Наблюдаемость, конфигурация и безопасность

### Логи и наблюдаемость

Логируются все стадии пайплайна через `logging.config.dictConfig` (настройка в `src/main.py`).  
Уровень логов задаётся через `LOG_LEVEL` в `.env` (по умолчанию `INFO`).

Логируются:
- загрузка и предобработка аудио;
- стадии пайплайна (ASR, diarization, extraction);
- выбор модели (`provider`), стадия разбора (`parse_stage`), флаг fallback (`fallback_used`);
- ошибки API и модели;
- статус БД при каждом вызове `/api/health`.

### Конфигурация

Все параметры вынесены в `.env` (шаблон: `.env.example`):
- Whisper: `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`
- Diarization: `DIARIZATION_ENABLED`, `HF_TOKEN`
- Task extraction: `TASK_PROVIDER`, `OPENROUTER_API_KEY`, `OPENROUTER_TASK_MODEL`
- БД: `DATABASE_URL`
- Сервис: `LOG_LEVEL`, `MAX_UPLOAD_SIZE_MB`, `CORS_ORIGINS`

### Безопасность

- Реальные токены и пароли не хранятся в репозитории (`.env` в `.gitignore`).
- В репозитории — только `.env.example` с пустыми значениями.
- Политика управления секретами описана в `SECURITY.md`.
- Без API-ключей сервис работает в offline-режиме: Whisper + rule-based (без интернета).

---

## 8. Ограничения и дальнейшая работа

Ограничения проекта:
- diarization может ошибаться на шумных или очень коротких файлах;
- извлечение задач зависит от качества ответа LLM (6/13 встреч → parse_failed);
- отдельные модели OpenRouter ограничены rate limit;
- evaluation по gold не покрывает все возможные сценарии.

Дальнейшее развитие:
- few-shot примеры из AMI в system prompt (ожидаемый прирост F1 ≈ +0.05–0.10);
- fine-tuning `flan-t5-small` на AMI gold annotations;
- ансамблирование rule-based + LLM;
- Whisper `large-v3` для снижения WER до ~0.30 (требует GPU);
- расширение UI: ручная правка speaker-меток, расширенный dashboard.

---

## 9. Сценарий демонстрации на защите

```bash
# 1. Запуск через Docker
cp .env.example .env        # добавить OPENROUTER_API_KEY
docker build -t meeting-secretary .
docker run --rm -p 8000:8000 --env-file .env meeting-secretary

# 2. Тест ML endpoint (основной /predict)
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Alice will prepare the design spec by Friday. Bob needs to review requirements.", "language": "en"}'

# 3. Полный пайплайн (аудио → задачи)
curl -X POST http://localhost:8000/api/upload_meeting \
  -F "file=@data/audio/ES2002a.Mix-Headset.wav"

# 4. Метрики и оценка
curl http://localhost:8000/api/metrics
curl http://localhost:8000/api/evaluations
```

На защите планируется показать:
1. Swagger UI (`/docs`) — демо `/predict` и `/upload_meeting`;
2. Ноутбуки: `01_eda.ipynb`, `02_baselines.ipynb`;
3. Артефакты: `artifacts/model_comparison.csv`;
4. React UI — встречи, metrics, evaluation по gold.

---

## 10. Заключение

Проект вырос из линейного прототипа в экспериментальную платформу для анализа совещаний.  
В системе реализованы: транскрипция, diarization, speaker aliasing, извлечение задач, назначение исполнителей, evaluation по gold, сравнение моделей и аналитический интерфейс.

**Ключевые результаты:**
- Task F1: **0.257** (LLM) vs **0.140** (rule-based baseline) — улучшение **+83%**
- WER: **0.514** на AMI Meeting Corpus (Whisper small, CPU)
- `/predict` endpoint с автоматическим fallback — работает без API-ключей
- 3 ноутбука: EDA, сравнение baseline/LLM, тестирование endpoint
- 7 тестов (`pytest tests/`) — все проходят
- Все 10 критериев чеклиста выполнены

Это делает проект пригодным как для защиты на курсе, так и для дальнейшего развития.
