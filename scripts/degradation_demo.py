# ruff: noqa: E402 -- direct script execution requires repository path bootstrap
"""Offline evidence of one/two/three source failures, with no HTTP or database."""

import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from researcher.concurrency.models import SOURCE_NAMES
from researcher.concurrency.orchestrator import SourceOrchestrator
from researcher.concurrency.research import NoSourcesError, ResearchOrchestrator
from scripts.c_offline import OfflineHistory, OfflineService, offline_client


async def main() -> None:
    print("OFFLINE SIMULATION: fake providers, synthesis and history; no real HTTP/DB.")
    for count in (1, 2, 3):
        service, history = OfflineService(SOURCE_NAMES[:count]), OfflineHistory()
        async with offline_client() as client:
            collector = SourceOrchestrator(
                service, client=client, timeout_seconds=1, max_concurrency=3
            )
            runner = ResearchOrchestrator(collector, service, history)
            try:
                result = await runner.research("Demonstrate partial failure")
                print(
                    f"{count} failed: used={result.bundle.used}; failed={result.bundle.failed}; history={result.history_status}; answer={result.answer.answer}"
                )
            except NoSourcesError as exc:
                print(
                    f"{count} failed: NoSourcesError; failed={exc.collection.failed}; synthesis_calls={service.synthesis_calls}; history_rows={len(history.records)}"
                )


if __name__ == "__main__":
    asyncio.run(main())
