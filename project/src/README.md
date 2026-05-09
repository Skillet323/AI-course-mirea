# Source code

Основной код проекта.

## Ожидаемая структура

- `config.py` — настройки и переменные окружения.
- `db.py` — подключение к БД.
- `main.py` — FastAPI-приложение.
- `models.py` — SQLModel-схемы.
- `api/` — маршруты.
- `services/` — ASR, diarization, extraction, assignment, evaluation.
- `training/` — инструменты для экспериментов и сравнения моделей.
- `utils/` — вспомогательные функции.

## Примечание

Поскольку frontend и backend запускаются из одного Docker-образа, в `main.py` стоит:
- включить роуты под `/api`;
- раздавать `frontend/dist` как статику;
- возвращать `index.html` для SPA-переходов.
