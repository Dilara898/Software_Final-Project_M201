# ruff: noqa: E402 -- direct script execution requires repository path bootstrap
"""Collection benchmark with explicit offline or injected live services."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from collections.abc import Callable
from typing import cast


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from researcher.concurrency.benchmarking import (
    BenchmarkOptions,
    QuestionSet,
    run_benchmark,
)
from researcher.concurrency.contracts import FetchService
from researcher.services.http_client import source_client
from scripts.c_offline import OfflineService, offline_client


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_factory(spec: str) -> Callable[[], FetchService]:
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("service factory must be module:callable")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise ValueError("service factory must be callable")
    return cast(Callable[[], FetchService], factory)


def git_metadata() -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return f"{revision}; dirty={dirty}"
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument(
        "--service-factory",
        help="B-owned zero-argument factory module:callable; required for live",
    )
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/research_questions.json"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    if args.live and not args.service_factory:
        parser.error("--live requires --service-factory supplied by role B")
    if args.offline and args.service_factory:
        parser.error("--service-factory is only valid with --live")
    if args.out and args.csv and args.out.resolve() == args.csv.resolve():
        parser.error("summary and CSV must use different paths")
    if any(
        output is not None and output.resolve() == args.questions.resolve()
        for output in (args.out, args.csv)
    ):
        parser.error("output paths must not overwrite the questions dataset")
    try:
        raw = args.questions.read_bytes()
        questions = QuestionSet.model_validate_json(raw)
        options = BenchmarkOptions(
            repeats=args.repeats,
            concurrency=args.concurrency,
            source_timeout_seconds=args.timeout,
            max_results_per_source=args.max_results,
            warmup=args.warmup,
        )
        factory = load_factory(args.service_factory) if args.live else OfflineService
    except (OSError, ValueError, ImportError, AttributeError) as exc:
        parser.error(
            f"cannot initialize benchmark ({type(exc).__name__}); check dataset, options and factory"
        )
    report = asyncio.run(
        run_benchmark(
            questions,
            options,
            service_factory=factory,
            client_factory=(
                lambda: source_client(
                    timeout_seconds=options.source_timeout_seconds
                )
            )
            if args.live
            else offline_client,
        )
    )
    label = (
        "LIVE COLLECTION"
        if args.live
        else "OFFLINE SIMULATION — no real provider timings"
    )
    command = subprocess.list2cmdline(
        [
            sys.executable,
            "scripts/benchmark.py",
            *(sys.argv[1:] if argv is None else argv),
        ]
    )
    summary = report.markdown(label) + (
        f"\nUTC: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Runtime: Python {platform.python_version()}, {platform.system()}\n\n"
        f"Dataset SHA256: {hashlib.sha256(raw).hexdigest()}\n\nGit: {git_metadata()}\n\n"
        f"Service factory: {args.service_factory or 'scripts.c_offline:OfflineService'}\n\n"
        f"Reproduce: `{command}`\n"
    )
    try:
        if args.out:
            atomic_write(args.out, summary)
        if args.csv:
            atomic_write(args.csv, report.csv_text())
    except OSError:
        print(
            "Cannot write benchmark artifact; check output paths and permissions.",
            file=sys.stderr,
        )
        return 2
    print(summary)
    return 0 if len(report.comparable_repeats) == options.repeats else 2


if __name__ == "__main__":
    raise SystemExit(main())
