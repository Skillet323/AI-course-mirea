# syntax=docker/dockerfile:1.7

FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/project

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/project/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src /app/project/src
COPY data /app/project/data
COPY configs /app/project/configs
COPY scripts /app/project/scripts
COPY tests /app/project/tests
COPY artifacts /app/project/artifacts
COPY README.md /app/project/README.md
COPY report.md /app/project/report.md
COPY self-checklist.md /app/project/self-checklist.md
COPY .env.example /app/project/.env.example

COPY --from=frontend-build /app/frontend/dist /app/project/frontend/dist

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
