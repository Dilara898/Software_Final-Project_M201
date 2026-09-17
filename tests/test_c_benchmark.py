import csv
import io
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from ai.schemas import Source
from researcher.concurrency.benchmarking import (
    BenchmarkOptions,
    QuestionSet,
    run_benchmark,
)
from scripts.benchmark import atomic_write, load_factory, main
from scripts.c_offline import offline_client


@pytest.mark.parametrize(
    "payload",
    [
        {"questions": []},
        {"questions": [{"id": "a", "text": " "}]},
        {"questions": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]},
    ],
)
def test_dataset_rejects_invalid_workloads(payload):
    with pytest.raises(ValidationError):
        QuestionSet.model_validate(payload)


@pytest.mark.parametrize(
    "options",
    [{"repeats": 0}, {"concurrency": True}, {"source_timeout_seconds": float("nan")}],
)
def test_invalid_options(options):
    with pytest.raises(ValidationError):
        BenchmarkOptions(**options)


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
async def test_raw_measurements_and_warmup(empty):
    services = []

    def factory():
        service = AsyncMock()

        async def fetch(source, query, client):
            return (
                []
                if empty
                else [
                    Source(
                        title=source,
                        url=f"https://example.invalid/{source}",
                        snippet=query,
                        origin=source,
                    )
                ]
            )

        service.fetch.side_effect = fetch
        services.append(service)
        return service

    report = await run_benchmark(
        QuestionSet(questions=[{"id": "q1", "text": "question"}]),
        BenchmarkOptions(repeats=2, warmup=True),
        service_factory=factory,
        client_factory=offline_client,
    )
    assert len(services) == 6 and all(s.fetch.await_count == 3 for s in services)
    assert [(b.repeat, b.mode) for b in report.batches] == [
        (1, "sequential"),
        (1, "parallel"),
        (2, "parallel"),
        (2, "sequential"),
    ]
    rows = list(csv.DictReader(io.StringIO(report.csv_text())))
    assert len(rows) == 12
    assert all(row["cache_status"] == "bypass" for row in rows)
    assert all(row["status"] == ("empty" if empty else "ok") for row in rows)
    assert report.comparable_repeats == ([] if empty else [1, 2])
    if empty:
        assert report.speedup is None
        assert "unavailable" in report.markdown("test")
    else:
        assert report.speedup == report.median_ms("sequential") / report.median_ms(
            "parallel"
        )


def test_atomic_output(tmp_path):
    target = tmp_path / "nested" / "report.md"
    atomic_write(target, "first")
    atomic_write(target, "second")
    assert target.read_text() == "second"
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize(
    "args",
    [
        ["--live"],
        ["--offline", "--repeats", "0"],
        ["--offline", "--service-factory", "x:y"],
        ["--live", "--service-factory", "missing_c_service_xyz:create"],
    ],
)
def test_cli_rejects_unusable_configuration(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_help_requires_no_service(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--live" in capsys.readouterr().out


def test_factory_validation():
    with pytest.raises(ValueError):
        load_factory("invalid")
    assert callable(load_factory("scripts.c_offline:OfflineService"))
