from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .api.evaluation import router as evaluation_router
from .api.routes import router as core_router
from .db import init_db

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": LOG_LEVEL,
        "handlers": ["console"],
    },
    "loggers": {
        "uvicorn": {"level": "INFO", "propagate": True},
        "uvicorn.access": {"level": "WARNING", "propagate": True},
    },
})

logger = logging.getLogger(__name__)


class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method == "POST" and "content-length" in request.headers:
            content_length = int(request.headers["content-length"])
            if content_length > self.max_upload_size:
                return Response("File too large", status_code=413)
        return await call_next(request)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan handler (replaces deprecated on_event)."""
    init_db()
    logger.info("Meeting Secretary started. LOG_LEVEL=%s", LOG_LEVEL)
    logger.info("API docs available at /docs")
    yield
    logger.info("Meeting Secretary shutting down.")


app = FastAPI(title="Meeting Secretary", lifespan=lifespan)

app.add_middleware(LimitUploadSize, max_upload_size=2 * 1024 * 1024 * 1024)

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.app\.github\.dev",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API должен быть именно под /api
app.include_router(core_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")


# Фронтенд из Docker / локальной сборки
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"status": "ok", "service": "meeting-secretary"}