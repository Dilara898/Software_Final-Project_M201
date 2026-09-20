"""Logging setup for Part B (and anything else that wants it).

Addresses the review's B2 gap: a configured `LOG_LEVEL`, a formatter that
actually renders the `extra={...}` fields every log call in `retry.py` /
`ai_service.py` / `transport.py` passes (Python's default `Formatter`
silently drops them unless the format string names each key), httpx/httpcore
noise suppression (both are otherwise very chatty at INFO/DEBUG), and
idempotent setup so importing this module twice -- or calling
`configure_logging()` twice -- never attaches a duplicate handler.

This module never itself decides *whether* to log secrets: every log call
in this package already follows the "codes, not exception text" discipline
documented in `retry.py`. This module only decides how a log record is
rendered to output.
"""

from __future__ import annotations

import logging
import os

# The set of LogRecord attribute names that exist on every record
# regardless of what the caller passed via `extra=`. Computed once from a
# throwaway record rather than hardcoded, so it stays correct if Python
# ever adds a new standard attribute.
_STANDARD_LOG_RECORD_ATTRS: frozenset[str] = frozenset(
    logging.LogRecord(
        name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
    ).__dict__.keys()
) | {"message", "asctime"}

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

# Third-party loggers that are useful at DEBUG but too noisy to inherit our
# own configured level by default (httpx/httpcore log every request at INFO
# and every low-level event at DEBUG).
_NOISY_THIRD_PARTY_LOGGERS: tuple[str, ...] = ("httpx", "httpcore")
_NOISY_THIRD_PARTY_FLOOR = logging.WARNING

_configured = False


class ExtraFieldsFormatter(logging.Formatter):
    """A `Formatter` that appends any `extra={...}` fields a log call
    passed, as `key=value` pairs, after the base formatted message.

    Without this, `log.warning("fetch_exhausted", extra={"source": "web",
    "error_code": "provider_error"})` renders as just `"fetch_exhausted"` --
    the whole point of passing structured `extra` fields is lost with the
    standard library's default `Formatter`.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_ATTRS and not key.startswith("_")
        }
        if not extras:
            return base
        extras_str = " ".join(f"{key}={value!r}" for key, value in sorted(extras.items()))
        return f"{base} | {extras_str}"


def configure_logging(
    *,
    level: str | None = None,
    stream_format: str = _DEFAULT_FORMAT,
    force: bool = False,
) -> None:
    """Configure the root logger once (idempotent unless `force=True`).

    Parameters
    ----------
    level:
        Log level name (e.g. "INFO", "DEBUG"). Defaults to the `LOG_LEVEL`
        environment variable (matching A's `researcher.config.Settings
        .log_level`), falling back to "INFO" if unset.
    force:
        Re-run setup even if already configured, replacing the existing
        handler(s) rather than adding another. Without this, calling
        `configure_logging()` a second time (e.g. from an importing module
        that also calls it) is a safe no-op -- never a duplicate handler
        producing every log line twice.
    """
    global _configured
    if _configured and not force:
        return

    resolved_level_name = level if level is not None else os.getenv("LOG_LEVEL", "INFO")
    resolved_level = getattr(logging, resolved_level_name.upper(), logging.INFO)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(ExtraFieldsFormatter(stream_format))
    root.addHandler(handler)
    root.setLevel(resolved_level)

    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(max(resolved_level, _NOISY_THIRD_PARTY_FLOOR))

    _configured = True


def reset_for_tests() -> None:
    """Undo `configure_logging`'s idempotency guard, for test isolation.

    Does not itself remove handlers; a subsequent `configure_logging()`
    call will do that. Only intended for use in tests that need to call
    `configure_logging` more than once and observe each call's effect.
    """
    global _configured
    _configured = False


__all__ = ["ExtraFieldsFormatter", "configure_logging", "reset_for_tests"]