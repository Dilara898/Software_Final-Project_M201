FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip check

RUN useradd --create-home appuser

COPY --chown=appuser:appuser researcher/ ./researcher/
COPY --chown=appuser:appuser ai/ ./ai/
COPY --chown=appuser:appuser data/ ./data/

USER appuser

# Temporary command until the full CLI is connected.
CMD ["python", "-c", "from researcher.cli import build_parser; build_parser().print_help()"]