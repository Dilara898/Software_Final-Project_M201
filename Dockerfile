FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip check

RUN useradd --create-home appuser \
    && mkdir -p /app/artefacts \
    && chown appuser:appuser /app/artefacts

COPY --chown=appuser:appuser researcher/ ./researcher/
COPY --chown=appuser:appuser ai/ ./ai/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser scripts/ ./scripts/

USER appuser

ENTRYPOINT ["python", "-m", "researcher"]
CMD ["--help"]