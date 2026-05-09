# Patch for `src/main.py`

Чтобы единый Dockerfile корректно запускал и backend, и frontend, в `src/main.py` нужно сделать два изменения:

## 1. Повесить API под `/api`

Заменить:

```python
app.include_router(core_router)
app.include_router(evaluation_router)
```

на:

```python
app.include_router(core_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")
```

## 2. Раздавать собранный frontend из `frontend/dist`

Добавить:

```python
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
```

И после подключения роутеров:

```python
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    def root():
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
```

Такой вариант оставляет `/docs` и `/api/*` рабочими, а UI открывается на `/`.
