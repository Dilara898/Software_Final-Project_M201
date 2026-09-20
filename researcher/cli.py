"""Command-line interface for the research assistant."""

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
        help="Skip cache reads and writes",
    )

    history = sub.add_parser("history", help="Show past queries")
    history.add_argument("--limit", type=int, default=10)

    return parser


async def show_history(limit: int) -> int:
    import asyncio
    import logging
    import sys

    import asyncpg
    from pydantic import ValidationError as SettingsError

    from researcher.storage.db import close_pool, get_pool
    from researcher.storage.history import list_sessions

    log = logging.getLogger(__name__)

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

    except SettingsError:
        print(
            "Configuration error: check DATABASE_URL and .env settings.",
            file=sys.stderr,
        )
        return 1

    except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
        log.error("History failed (%s)", type(exc).__name__)
        print(
            "Cannot read history. Check the database connection and tables.",
            file=sys.stderr,
        )
        return 1

    finally:
        try:
            await asyncio.wait_for(close_pool(), timeout=5)
        except Exception as exc:
            # Cleanup errors must not replace the original result.
            # Cancellation still propagates.
            log.warning(
                "Database cleanup failed (%s)",
                type(exc).__name__,
            )


def main(argv: list[str] | None = None) -> int:
    import asyncio
    import logging
    import os
    import sys

    # Parse help before loading configuration or opening resources.
    args = build_parser().parse_args(argv)

    from researcher.exceptions import ValidationError
    from researcher.validation import validate_limit, validate_question

    try:
        if args.command == "history":
            args.limit = validate_limit(args.limit)
        else:
            args.question = validate_question(args.question)

    except ValidationError as exc:
        print(f"researcher: {exc}", file=sys.stderr)
        return 1

    try:
        from dotenv import load_dotenv
        from pydantic import ValidationError as SettingsError

        from researcher.services.logging_setup import configure_logging

        # Existing process environment values take priority over .env.
        load_dotenv(".env", override=False)

        # Empty optional values should not override provider defaults.
        for name in ("LLM_MODEL", "LLM_PROVIDER", "WEB_SEARCH_PROVIDER"):
            if name in os.environ and not os.environ[name].strip():
                os.environ.pop(name)

        os.environ.setdefault("LLM_PROVIDER", "gemini")
        os.environ.setdefault("WEB_SEARCH_PROVIDER", "tavily")

        configure_logging()

        try:
            if args.command == "history":
                return asyncio.run(show_history(args.limit))

            from ai.providers.base import ProviderError
            from researcher.concurrency.research import OrchestrationError
            from researcher.exceptions import ResearcherError
            from researcher.application import run_ask

            try:
                return asyncio.run(run_ask(args))

            except (ResearcherError, OrchestrationError, ProviderError) as exc:
                logging.getLogger(__name__).error(
                    "Research failed (%s)",
                    type(exc).__name__,
                )
                print(
                    "Research failed. Check provider settings, "
                    "available sources and connectivity.",
                    file=sys.stderr,
                )
                return 1

        except SettingsError:
            print(
                "Invalid configuration. Check DATABASE_URL and .env settings.",
                file=sys.stderr,
            )
            return 1

    except ModuleNotFoundError as exc:
        print(
            f"Missing dependency: {exc.name}. "
            "Install the project requirements.",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130