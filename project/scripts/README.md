# Scripts

Скрипты для пакетной обработки, подготовки gold-разметки и сравнения моделей.

## Основные скрипты

- `build_ami_gold.py` — сборка pseudo-gold JSON из AMI аудио/транскриптов.
- `import_gold_annotations.py` — импорт gold в БД.
- `compare_task_models.py` — сравнение моделей извлечения задач.
- `upload_audio_queue.py` — очередная загрузка аудио в backend.

## Как использовать

Запускать из корня `project/`, чтобы пути к `data/`, `artifacts/` и `src/` совпадали с документацией.
