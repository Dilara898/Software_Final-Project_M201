import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researcher",
        description="Builds a cited answer from Wikipedia, arXiv, and web search.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask a question and get a cited answer")
    ask.add_argument("question", help="The research question")
    ask.add_argument(
        "--sources",
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