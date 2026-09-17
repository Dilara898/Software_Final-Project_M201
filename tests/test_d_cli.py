import pytest

from researcher.cli import build_parser


def test_ask_defaults():
    args = build_parser().parse_args([
        "ask", "What is photosynthesis?",
    ])

    assert args.command == "ask"
    assert args.question == "What is photosynthesis?"
    assert args.sources is None
    assert args.no_cache is False


def test_ask_with_options():
    args = build_parser().parse_args([
        "ask",
        "What is photosynthesis?",
        "--sources",
        "wiki,arxiv",
        "--no-cache",
    ])

    assert args.sources == "wiki,arxiv"
    assert args.no_cache is True


def test_history_default_limit():
    args = build_parser().parse_args(["history"])

    assert args.command == "history"
    assert args.limit == 10


def test_history_custom_limit():
    args = build_parser().parse_args(["history", "--limit", "5"])

    assert args.limit == 5


def test_ask_requires_question():
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["ask"])

    assert error.value.code == 2


def test_history_rejects_non_integer_limit():
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["history", "--limit", "abc"])

    assert error.value.code == 2


@pytest.mark.parametrize("sources", [
    "wiki", "arxiv", "web", "wiki,arxiv,web",
])
def test_sources_accepts_valid_values(sources):
    args = build_parser().parse_args([
        "ask", "What is photosynthesis?", "--sources", sources,
    ])

    assert args.sources == sources


def test_sources_normalizes_input():
    args = build_parser().parse_args([
        "ask",
        "What is photosynthesis?",
        "--sources",
        " Wiki, arxiv,wiki ",
    ])

    assert args.sources == "wiki,arxiv"


@pytest.mark.parametrize("sources", [
    "", " ", "google", "wiki,google", "wiki,", ",wiki", "wiki,,arxiv",
])
def test_sources_rejects_invalid_values(sources, capsys):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([
            "ask", "What is photosynthesis?", "--sources", sources,
        ])

    assert error.value.code == 2
    assert "argument --sources:" in capsys.readouterr().err