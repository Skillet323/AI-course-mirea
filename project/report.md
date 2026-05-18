# Отчёт по итоговому проекту по курсу «Инженерия Искусственного Интеллекта»

---

## 1. Паспорт проекта

- **Название проекта:** `Meeting Secretary`
- **Автор:** `Боргачев Тимофей Максимович`
- **Группа:** `ИНБО-20-23`
- **Контакт:** `@Skillet323`
- **Ссылка на репозиторий:** `https://github.com/Skillet323/AI-course-mirea`

Проект — веб-сервис для автоматической обработки аудиозаписей совещаний.
Система распознаёт речь, выполняет diarization, извлекает поручения, сопоставляет их с участниками, сохраняет результаты и позволяет сравнивать конфигурации моделей по метрикам.

---

## 2. Постановка задачи и контекст

Проект решает задачу автоматической обработки аудиозаписей совещаний. Пользователь загружает запись встречи, после чего система:

- преобразует аудио в текст;
- выделяет поручения и action items;
- определяет ответственных;
- сохраняет результат в базе данных;
- показывает метрики и диагностические признаки обработки.

Потенциальный пользователь — сотрудник или аналитик, которому важно быстро получить структурированный итог совещания без ручного протоколирования.

### Формулировка задачи в терминах ML/ИИ

- **ASR**: преобразование аудио в текст (Whisper medium);
- **speaker diarization**: разделение реплик по спикерам (pyannote → ECAPA-TDNN → resemblyzer);
- **information extraction**: извлечение задач из транскрипта (LLM + rule-based fallback);
- **entity/role matching**: сопоставление задач с участниками (assignment engine).

### Целевые метрики качества

- **WER** и **CER** — качество распознавания речи;
- **Task Precision**, **Task Recall**, **Task F1** — качество извлечения задач;
- **Assignee Accuracy** и **Deadline Accuracy** — точность атрибутов задач;
- **transcript_confidence** — внутренняя уверенность Whisper (вспомогательная);
- **latency** — время обработки по стадиям пайплайна.

---

## 3. Данные

### AMI Meeting Corpus (benchmark / eval)

- Публичный датасет **AMI Meeting Corpus**, серия ES (открытая лицензия).
- 13 встреч (`ES2002a–ES2014a`), продолжительность 5–23 минуты.
- Gold-аннотации: 37 задач (≈ 2.85 на встречу), ~77% с `assignee_hint`, ~69% с `deadline_hint`.

### Реальные встречи (production)

По состоянию на 18.05.2026 через систему прошло **16 встреч** с суммарно **141 задачей**:

| Показатель | Значение |
|---|---:|
| Всего встреч | 16 |
| Всего задач | 141 |
| Среднее задач / встречу | 9.4 |
| Средняя уверенность транскрипта | 0.809 |
| Среднее время обработки | 246.8 с |
| Языки (en / cy / nn) | 13 / 1 / 1 |
| Модель Whisper | medium (все встречи) |

Время обработки: транскрипция ~93–95% (≈ 232 с), извлечение задач ~4–14 с, назначение исполнителей < 0.01 с.

### Структура данных

- `data/audio/` — аудиозаписи;
- `data/gold/` — эталонные аннотации AMI в JSON;
- `artifacts/` — результаты сравнения моделей.

### Предобработка и EDA

Подробный EDA — в `notebooks/01_eda.ipynb`.

Выполнялись: нормализация аудио, транскрибация через Whisper medium, анализ WER/CER по gold, сравнение с gold-разметкой, анализ распределения задач и лексического покрытия.

Ключевые наблюдения:
- WER варьируется от 0.30 до 0.75 в зависимости от качества записи;
- среднее лексическое покрытие задачи транскриптом — ~55%;
- поручения часто сформулированы неявно → rule-based baseline имеет низкий precision.

---

## 4. Модели и подходы

### Компоненты системы

| Компонент | Реализация |
|---|---|
| ASR | Whisper **medium** (CPU/GPU), openai-whisper |
| Diarization (primary) | pyannote/speaker-diarization-3.1, чанкование произвольной длины |
| Diarization (intermediate) | ECAPA-TDNN / SpeechBrain |
| Diarization (fallback) | resemblyzer + silhouette-optimal AgglomerativeClustering |
| Task extraction (production) | `google/gemini-2.5-flash-lite`, `google/gemini-3-flash-preview` |
| Task extraction (fallback) | rule-based (40+ action-маркеров) |
| Task classifier (ML) | sklearn pipeline (`artifacts/task_sentence_classifier.pkl`) |

### Ключевые улучшения над baseline

- recursive parser для ответа LLM (устойчивость к нестандартному JSON);
- conservative mode для коротких/шумных записей;
- проверка согласованности задачи с транскриптом (`_task_supported_by_transcript`);
- speaker aliases для замены `SPEAKER_XX` на имена;
- global greedy matching в evaluation (матрица сходств);
- diarization без жёсткого лимита по длительности — автоматическое чанкование;
- silhouette-based поиск числа спикеров; AgglomerativeClustering (Ward);
- assignment работает без зарегистрированных участников (из `assignee_hint` / snippet).

---

## 5. Экспериментальный протокол и результаты

Все benchmark-эксперименты на 13 встречах AMI. Артефакты: `artifacts/model_comparison.{csv,json}`, `artifacts/nvidia_models_comparison.{csv,json}`, `artifacts/openrouter_free.{csv,json}`. Анализ — в `notebooks/02_baselines.ipynb`.

### Сравнение моделей: OpenRouter (основной бенчмарк)

| Модель | OK/всего | Task F1 | Precision | Recall | Assignee | Deadline | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Rule-based baseline** | 13/13 | 0.140 | 0.080 | 0.628 | — | — | ~0 с |
| `inclusionai/ling-2.6-1t:free` | 7/13 | **0.257** | **0.244** | **0.286** | 0.125 | 0.250 | 4.6 с |
| `openai/gpt-oss-120b:free` | 3/13 | 0.222 | 0.167 | 0.333 | 0.500 | 0.000 | 17.2 с |
| `poolside/laguna-m.1:free` | 1/13 | — | — | — | — | — | 22.2 с |

### Масштабное сравнение OpenRouter free-tier (27 моделей, 18.05.2026)

Прогон 27 моделей на 13 встречах AMI (`artifacts/openrouter_free.csv`). 14 из 27 недоступны из-за rate limit. Среди успешных (все 13 встреч):

| Модель | OK/всего | Task F1 | Assignee | Latency |
|---|---:|---:|---:|---:|
| `liquid/lfm-2.5-1.2b-instruct:free` | 5/13* | 0.129 | 0.500 | 7.8 с |
| `openrouter/owl-alpha` | 13/13 | 0.099 | 0.250 | 11.3 с |
| `openai/gpt-oss-120b:free` | 13/13 | 0.031 | 0.000 | 7.0 с |
| `arcee-ai/trinity-large-thinking:free` | 13/13 | 0.009 | 0.000 | 5.6 с |

\* остальные 8 встреч — rate limit.

**Вывод:** free-tier OpenRouter в мае 2026 нестабилен. `openrouter/owl-alpha` (0.099 F1, 13/13) — лучший по надёжности, но ниже основного бенчмарка. Используется `google/gemini-2.5-flash-lite` на платном tier.

### Сравнение моделей NVIDIA Build API

| Модель | OK/всего | Task F1 | Precision | Recall | Latency |
|---|---:|---:|---:|---:|---:|
| `upstage/solar-10.7b-instruct` | 7/13 | **0.212** | 0.181 | 0.262 | 12.1 с |
| `google/gemma-3n-e4b-it` | **13/13** | 0.203 | 0.162 | **0.282** | 25.9 с |
| `google/gemma-3n-e2b-it` | **13/13** | 0.185 | 0.142 | 0.308 | 29.2 с |
| `qwen/qwen3-coder-480b-a35b-instruct` | 13/13 | 0.144 | 0.154 | 0.154 | 10.8 с |
| `mistralai/mistral-large-3-675b-instruct-2512` | 13/13 | 0.097 | 0.094 | 0.128 | 89.3 с |
| `nvidia/nemotron-mini-4b-instruct` | 12/13 | 0.000 | 0.000 | 0.000 | 0.3 с |

**Вывод по NVIDIA:** `google/gemma-3n-e4b-it` — лучший компромисс (F1=0.203, 13/13, умеренная latency). Рекомендован при `TASK_PROVIDER=nvidia`.

### Производительность в production (16 встреч, Whisper medium)

| Встреч | Задач | Задач/встречу | Confidence | Total latency |
|---:|---:|---:|---:|---:|
| 16 | 141 | 9.4 | 0.809 | 246.8 с |

Task-модели в production: `google/gemini-2.5-flash-lite` (основная, latency 4–14 с), `google/gemini-3-flash-preview-20251217` (альтернативная), `mistralai/mistral-7b-instruct-v0.1` (редкий fallback).

### Выбор финальной модели

**Production:** `google/gemini-2.5-flash-lite` (OpenRouter платный tier).

Обоснование: стабильна (16/16 встреч без ошибок), latency 4–14 с, отсутствие rate limit на платном tier, интегрирована в существующий OpenRouter-роутер.

**Benchmark (free OpenRouter):** `inclusionai/ling-2.6-1t:free` — наивысший F1=0.257, несмотря на нестабильность parse (6/13 parse_failed, автоматический fallback на rule-based).

**NVIDIA API:** `google/gemma-3n-e4b-it` — F1=0.203, 13/13 надёжность.

Rule-based baseline: используется только как fallback; высокий recall (0.63) компенсирует низкий precision при недоступности LLM.

---

## 6. Архитектура решения и сервис

Пайплайн:

```
Аудио → Preprocessing → ASR (Whisper medium)
      → Diarization (pyannote → ECAPA-TDNN → resemblyzer)
      → Speaker Transcript
      → Task Extraction (LLM → fallback: rule-based)
      → Assignment Engine
      → DB (SQLite) → Evaluation → React UI
```

### API endpoints

| Endpoint | Метод | Описание |
|---|---|---|
| `/api/predict` | POST | **ML inference**: `transcript` → `tasks` |
| `/api/upload_meeting` | POST | Полный пайплайн: аудиофайл → задачи |
| `/api/health` | GET | Статус сервиса + БД |
| `/api/meeting/{id}` | GET | Детали встречи |
| `/api/meeting/{id}/progress` | GET | Прогресс обработки (SSE) |
| `/api/meeting/{id}/export` | GET | Экспорт JSON/CSV |
| `/api/metrics` | GET | Метрики (latency, quality) |
| `/api/stats` | GET | Агрегированная статистика |
| `/api/evaluations` | GET | Результаты оценки по gold |

### Стек

- **Backend:** FastAPI + SQLModel + SQLite/PostgreSQL
- **ASR:** openai-whisper (medium, CPU/GPU)
- **Diarization:** pyannote.audio / SpeechBrain ECAPA-TDNN / resemblyzer
- **Task extraction:** OpenRouter API (Gemini) + rule-based fallback
- **Frontend:** React + Vite
- **Деплой:** Docker (multi-stage Node→Python), `Dockerfile.gpu`

---

## 7. Наблюдаемость, конфигурация и безопасность

### Наблюдаемость

- `logging.config.dictConfig` во всех модулях `src/`;
- логируются: стадии пайплайна, выбор модели, parse_stage, fallback_used, ошибки API;
- `GET /api/health` — статус БД;
- прогресс через SSE (`/api/meeting/{id}/progress`).

### Конфигурация (`.env` / `.env.example`)

- Whisper: `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`
- Diarization: `DIARIZATION_ENABLED`, `HF_TOKEN`, `PYANNOTE_CHUNK_SEC`
- Tasks: `TASK_PROVIDER`, `OPENROUTER_API_KEY`, `OPENROUTER_TASK_MODEL`
- БД: `DATABASE_URL`; Сервис: `LOG_LEVEL`, `MAX_UPLOAD_SIZE_MB`

### Безопасность

- `.env` в `.gitignore`; в репозитории только `.env.example` с пустыми значениями;
- без ключей — offline-режим (Whisper + rule-based);
- подробности — `SECURITY.md`.

---

## 8. Ограничения и дальнейшая работа

Ограничения:
- Whisper medium на CPU: ~230 с на 20-минутное аудио; GPU ускоряет в 10–15×;
- без HF_TOKEN — diarization fallback (resemblyzer, качество ниже);
- free-tier OpenRouter нестабилен (rate limit, parse_failed);
- evaluation по gold только для AMI; production-встречи не размечены.

Дальнейшее развитие:
- few-shot примеры из AMI в system prompt (ожидаемый прирост F1 ≈ +0.05–0.10);
- fine-tuning `flan-t5-small` на AMI gold;
- ансамблирование rule-based + LLM;
- Whisper `large-v3` (WER ~0.30, требует GPU);
- ручная правка speaker-меток в UI.

---

## 9. Сценарий демонстрации

```bash
# 1. Запуск через Docker
cp .env.example .env   # добавить OPENROUTER_API_KEY
docker build -t meeting-secretary .
docker run --rm -p 8000:8000 --env-file .env meeting-secretary

# 2. ML endpoint
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Alice will prepare the design spec by Friday.", "language": "en"}'

# 3. Полный пайплайн
curl -X POST http://localhost:8000/api/upload_meeting \
  -F "file=@data/audio/ES2002a.Mix-Headset.wav"

# 4. Метрики
curl http://localhost:8000/api/metrics
curl http://localhost:8000/api/evaluations
```

На защите: Swagger UI, ноутбуки 01_eda + 02_baselines, артефакты model_comparison + openrouter_free, React UI (16 встреч, metrics, evaluation).

---

## 10. Заключение

**Ключевые результаты:**

| Показатель | Значение |
|---|---|
| Task F1 (лучший LLM, benchmark) | **0.257** (`inclusionai/ling-2.6-1t:free`) |
| Task F1 (rule-based baseline) | 0.140 (+83% от LLM) |
| Production-встреч обработано | **16** |
| Production-задач извлечено | **141** (avg 9.4/встречу) |
| Avg confidence транскрипта | **0.809** |
| Модель ASR | Whisper **medium** |
| Моделей протестировано | **30+** (OpenRouter + NVIDIA) |
| Тесты pytest | 7 / 7 ✅ |
| Чеклист | 10 / 10 ✅ |
