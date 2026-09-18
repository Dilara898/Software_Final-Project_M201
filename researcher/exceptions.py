"""Shared exception family for the CLI/application boundary.

Owned by Part B, per external review finding B-01: `researcher/validation.py`
(D's input validation) does `from researcher.exceptions import
ValidationError`, but this module did not exist -- any import of
`researcher.validation` raised `ModuleNotFoundError: researcher.exceptions`.
This file exists to unblock that import with the minimal shared family the
review asked for: `ResearcherError`, `ValidationError`, `NoSourcesError`.

Relationship to C's own errors
-------------------------------
`researcher/concurrency/research.py` already defines `OrchestrationError`
(and its subclasses, including its own `NoSourcesError`), documented there
as "C's public error boundary; D can map it to the shared CLI error
family." That comment is the intended design: C's errors carry rich
diagnostics (e.g. its `NoSourcesError.collection`) and are not replaced or
subclassed from here -- doing so would mean editing `researcher/concurrency/
research.py`, which is out of scope for B without team agreement.

Instead, `NoSourcesError` in *this* module is the shared-family type D's
CLI boundary raises/displays generically (e.g. "no usable sources, try a
different query", exit code, etc.). The recommended composition, to be
confirmed with C/D rather than decided unilaterally here, is: D's CLI
catches `researcher.concurrency.research.NoSourcesError` (and C's other
`OrchestrationError` subclasses) at the call site and maps each to the
matching type in this module -- or re-raises the shared type with `from`
the original, so the diagnostic chain (`.collection`, etc.) is not lost,
just not part of this shared vocabulary. A formal `isinstance`/inheritance
relationship between the two `NoSourcesError` types was deliberately not
established here; see docs/ROLE_B.md for the open decision.
"""

from __future__ import annotations


class ResearcherError(Exception):
    """Base of the shared, CLI-facing exception family.

    Anything raised as a `ResearcherError` (or a subclass) is expected to
    reach the CLI boundary as a clean, user-facing message plus an exit
    code -- never a raw traceback or an exception message containing
    secrets (API keys, full request URLs). Callers that only need "did
    something in our own validated-input/expected-outcome space go wrong"
    can catch this base type without enumerating every subclass.
    """


class ValidationError(ResearcherError):
    """Input failed validation before any provider/network call was made.

    Raised by `researcher.validation` for CLI input (question text, history
    limit, source selection, etc.) that fails a structural check -- wrong
    length, wrong type, an unknown source name. Never raised for a
    provider/network failure; those are `OrchestrationError` (C) or
    `UpstreamDataError` (B/C's port contract), not this.
    """


class NoSourcesError(ResearcherError):
    """No usable sources were available to answer the question.

    The shared-family counterpart to
    `researcher.concurrency.research.NoSourcesError` -- see this module's
    docstring for how the two relate. Carries an optional `reason` for a
    short, secret-free explanation (e.g. "all sources timed out"); never
    put raw exception text or a full diagnostics object here that might
    leak into a CLI message uncontrolled.
    """

    def __init__(self, reason: str | None = None) -> None:
        super().__init__(reason or "No usable sources were found for this question.")
        self.reason = reason


__all__ = ["ResearcherError", "ValidationError", "NoSourcesError"]