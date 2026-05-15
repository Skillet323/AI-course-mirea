# Security Policy

## Управление секретами

В проекте используется схема конфигурации через переменные окружения.

### Что НЕ хранится в репозитории

- Реальные API-ключи (OpenRouter, HuggingFace и т.д.)
- Токены доступа
- Пароли к базам данных
- Любые production-секреты

### Как задать секреты

1. Скопируйте шаблон:
   ```bash
   cp .env.example .env
   ```
2. Заполните реальные значения в `.env`:
   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   HF_TOKEN=hf_...
   ```
3. Файл `.env` добавлен в `.gitignore` и не попадает в VCS.

### Шаблон `.env.example`

Файл `.env.example` содержит только пустые значения и описания переменных.  
Его безопасно коммитить в репозиторий.

### Получение токенов

- **OpenRouter API key**: https://openrouter.ai/keys (бесплатный tier доступен)
- **HuggingFace token**: https://huggingface.co/settings/tokens  
  После получения токена примите условия использования моделей:
  - https://huggingface.co/pyannote/speaker-diarization-3.1
  - https://huggingface.co/pyannote/segmentation-3.0

### Запуск без внешних API

Если API-ключи не заданы, сервис автоматически переключается на:
- **ASR**: Whisper (локально, без интернета)
- **Task extraction**: rule-based fallback (без OpenRouter)
- **Diarization**: resemblyzer (без HuggingFace token)

Это позволяет запускать проект в автономном режиме.

## Данные

- Данные в `data/gold/` — публичный датасет AMI Meeting Corpus (открытая лицензия).
- Персональные данные не хранятся.
- Audio-файлы не включены в репозиторий (только gold-аннотации в JSON).
