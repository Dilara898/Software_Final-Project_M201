import argparse


def validate_sources(value: str) -> str:
    allowed = {"wiki", "arxiv", "web"}
    sources = [source.strip().lower() for source in value.split(",")]

    if any(not source for source in sources):
        raise argparse.ArgumentTypeError(
            "Sources cannot be empty. Use: wiki,arxiv,web."
        )

    invalid = [source for source in sources if source not in allowed]

    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown sources: {', '.join(invalid)}. "
            "Allowed: wiki,arxiv,web."
        )

    return ",".join(dict.fromkeys(sources))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researcher",
        description="Builds a cited answer from Wikipedia, arXiv, and web search.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser(
        "ask",
        help="Ask a question and get a cited answer",
    )
    ask.add_argument("question", help="The research question")
    ask.add_argument(
        "--sources",
        type=validate_sources,
        default=None,
        help="Comma-separated: wiki,arxiv,web (default: all)",
    )
    ask.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip the cache",
    )

    history = sub.add_parser("history", help="Show past queries")
    history.add_argument("--limit", type=int, default=10)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    parser.exit(
        status=2,
        message=(
            f"researcher: '{args.command}' is not available yet. "
            "Service integration is still in progress.\n"
        ),
    )