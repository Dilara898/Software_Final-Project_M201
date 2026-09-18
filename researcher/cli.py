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


async def show_history(limit: int) -> int:
    import asyncio
    import sys

    import asyncpg
    from pydantic import ValidationError as SettingsError

    try:
        from researcher.storage.db import get_pool, close_pool
        from researcher.storage.history import list_sessions
    except SettingsError:
        print(
            "Configuration error: check DATABASE_URL in your .env file.",
            file=sys.stderr,
        )
        return 1

    try:
        try:
            pool = await asyncio.wait_for(get_pool(), timeout=10)
            sessions = await asyncio.wait_for(
                list_sessions(pool, limit=limit),
                timeout=10,
            )

            if not sessions:
                print("No previous queries found.")
                return 0

            for session in sessions:
                print(f"\n#{session.id} | {session.created_at}")
                print(f"Question: {session.question}")
                print(f"Answer: {session.answer}")

            return 0
        finally:
            await asyncio.wait_for(close_pool(), timeout=5)
    except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
        print(f"Error type: {type(exc).__name__}", file=sys.stderr)
        print(
            "Cannot read history. Check the database connection and tables.",
            file=sys.stderr,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    import asyncio
    import sys

    from researcher.exceptions import ValidationError
    from researcher.validation import validate_limit, validate_question

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "history":
            limit = validate_limit(args.limit)
            return asyncio.run(show_history(limit))

        args.question = validate_question(args.question)
    except ValidationError as exc:
        print(f"researcher: {exc}", file=sys.stderr)
        return 2
    except ModuleNotFoundError as exc:
        print(
            f"Missing dependency: {exc.name}. Install project requirements.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130

    print(
        "researcher: 'ask' is not connected to the research service yet.",
        file=sys.stderr,
    )
    return 2