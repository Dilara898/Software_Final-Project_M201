# ruff: noqa: E402 -- direct script execution requires repository path bootstrap
"""Run research questions through the application's existing ask workflow."""

import argparse
import asyncio
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

# Allow: python scripts/demo.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import asyncpg
import httpx
from dotenv import load_dotenv
from pydantic import ValidationError as SettingsError

from ai.providers.base import ProviderError
from researcher.application import execute_ask
from researcher.cli import validate_sources
from researcher.concurrency.research import OrchestrationError
from researcher.config import get_settings
from researcher.exceptions import ResearcherError
from researcher.services.ai_service import AIFetchService, AISynthesisService
from researcher.services.http_client import source_client
from researcher.services.logging_setup import configure_logging
from researcher.storage.db import close_pool, get_pool
from researcher.validation import validate_question

log = logging.getLogger(__name__)

EXPECTED_ERRORS = (
    ResearcherError,
    OrchestrationError,
    ProviderError,
    SettingsError,
    asyncpg.PostgresError,
    httpx.HTTPError,
    OSError,
    TimeoutError,
)


def load_questions(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("Dataset must be a JSON object.")

    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Dataset must contain a nonempty questions list.")

    validated = []
    seen = set()

    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Each question must be an object.")

        question_id = item.get("id")
        text = item.get("text")

        if (
            not isinstance(question_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", question_id)
            or question_id in seen
        ):
            raise ValueError("Question ids must be unique and filename-safe.")

        if not isinstance(text, str):
            raise ValueError("Question text must be a string.")

        validated.append(
            {
                "id": question_id,
                "text": validate_question(text),
            }
        )
        seen.add(question_id)

    return validated


async def run_questions(questions, args, output_dir):
    settings = get_settings()
    synthesis = None
    client = None
    outcomes = []

    try:
        pool = await asyncio.wait_for(get_pool(), timeout=10)

        synthesis = AISynthesisService()
        budget = settings.per_source_timeout_seconds

        fetch = AIFetchService(
            max_results=settings.max_sources_per_query,
            timeout_seconds=budget / 4,
            initial_wait_seconds=budget / 40,
            max_wait_seconds=budget / 20,
            min_interval_seconds=1.0,
        )

        client = source_client(timeout_seconds=budget)

        for item in questions:
            question_id = item["id"]
            print(f"Running {question_id}...", flush=True)

            question_args = SimpleNamespace(
                question=item["text"],
                sources=args.sources,
                no_cache=args.no_cache,
            )

            captured = io.StringIO()

            try:
                # Reuse D's actual application workflow and renderer.
                # Logging continues to stderr.
                with redirect_stdout(captured):
                    exit_code = await execute_ask(
                        question_args,
                        settings,
                        pool,
                        client,
                        fetch,
                        synthesis,
                    )

                status = "completed" if exit_code == 0 else "failed"
                body = captured.getvalue()

            except EXPECTED_ERRORS as exc:
                status = "failed"
                error_type = type(exc).__name__

                log.error(
                    "Demo question failed (%s)",
                    error_type,
                )

                body = (
                    f"Question: {item['text']}\n"
                    f"FAILED: {error_type}\n"
                    "Check configuration, provider availability "
                    "and database connectivity.\n"
                )

            artifact = (
                "Mode: LIVE\n"
                f"Question ID: {question_id}\n"
                f"Sources: {args.sources or 'wiki,arxiv,web'}\n"
                f"Status: {status}\n"
                "Note: completed means the application returned success; "
                "answer completeness still requires review.\n\n"
                + body
            )

            (output_dir / f"{question_id}.txt").write_text(
                artifact,
                encoding="utf-8",
            )

            outcomes.append(
                {
                    "id": question_id,
                    "status": status,
                }
            )

            print(f"{question_id}: {status}", flush=True)

        return outcomes

    finally:
        if synthesis is not None:
            try:
                synthesis.shutdown()
            except Exception as exc:
                log.warning(
                    "Synthesis cleanup failed (%s)",
                    type(exc).__name__,
                )

        if client is not None:
            try:
                await asyncio.wait_for(client.aclose(), timeout=5)
            except Exception as exc:
                log.warning(
                    "HTTP cleanup failed (%s)",
                    type(exc).__name__,
                )

        try:
            await asyncio.wait_for(close_pool(), timeout=5)
        except Exception as exc:
            log.warning(
                "Database cleanup failed (%s)",
                type(exc).__name__,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "research_questions.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "artefacts",
    )
    parser.add_argument(
        "--sources",
        type=validate_sources,
        default=None,
        help="Comma-separated: wiki,arxiv,web (default: all)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip cache reads and writes",
    )
    args = parser.parse_args(argv)

    try:
        questions = load_questions(args.questions)
    except (OSError, ValueError, ResearcherError):
        print(
            "Invalid dataset. Check the file, question ids and text.",
            file=sys.stderr,
        )
        return 1

    load_dotenv(PROJECT_ROOT / ".env", override=False)

    for name in ("LLM_MODEL", "LLM_PROVIDER", "WEB_SEARCH_PROVIDER"):
        if name in os.environ and not os.environ[name].strip():
            os.environ.pop(name)

    os.environ.setdefault("LLM_PROVIDER", "gemini")
    os.environ.setdefault("WEB_SEARCH_PROVIDER", "tavily")
    configure_logging()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.out / f"live-{stamp}"

    metadata = {
        "mode": "LIVE",
        "started_utc": stamp,
        "provider": os.getenv("LLM_PROVIDER"),
        "model": os.getenv("LLM_MODEL", "provider default"),
        "sources": args.sources or "wiki,arxiv,web",
        "use_cache": not args.no_cache,
        "expected_sources_policy": "metadata only, not a filter",
        "status": "running",
    }

    try:
        # A new directory prevents old results from looking like a new run.
        output_dir.mkdir(parents=True, exist_ok=False)
        manifest = output_dir / "run.json"

        def save_manifest():
            manifest.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        save_manifest()

        try:
            outcomes = asyncio.run(
                run_questions(questions, args, output_dir)
            )
        except BaseException:
            metadata["status"] = "aborted"
            save_manifest()
            raise

        failed = sum(
            item["status"] == "failed"
            for item in outcomes
        )

        metadata.update(
            status="completed",
            total=len(questions),
            failed=failed,
            results=outcomes,
        )
        save_manifest()

        print(
            f"\nCompleted: {len(questions) - failed}/{len(questions)}"
        )
        print(f"Output: {output_dir}")

        return 1 if failed else 0

    except EXPECTED_ERRORS as exc:
        print(
            f"Demo failed ({type(exc).__name__}). "
            "Check configuration, services and output permissions.",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print("\nDemo cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())