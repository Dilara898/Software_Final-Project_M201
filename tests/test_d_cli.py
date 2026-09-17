import pytest

from researcher.cli import build_parser


def test_ask_defaults():
    args = build_parser().parse_args(["ask", "Fotosintez nedir?"])

    assert args.command == "ask"
    assert args.question == "Fotosintez nedir?"
    assert args.sources is None
    assert args.no_cache is False


def test_ask_with_options():
    args = build_parser().parse_args([
        "ask",
        "Fotosintez nedir?",
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