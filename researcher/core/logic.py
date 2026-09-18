"""Source selection (alias/dedup/validation) and reference formatting.

This module is deliberately thin and provider/network-free: it does not
call any AI provider, fetcher, or async port. It exists for D's CLI (or
anything else) to turn a raw `--sources` argument into the canonical list
`FetchService`/`SourceOrchestrator` expect, and to render an
`AnswerWithCitations` for display -- both pure data transformations over
already-validated types.

Per the external review's own guidance: C already collects sources in a
fixed canonical order and `ai.synthesizer.synthesize` already builds
`AnswerWithCitations.citations` from only the indices actually cited in the
answer text (not the full candidate list). This module does not
reimplement either of those algorithms; `used_references` below is a thin
read/formatting adapter over what `synthesize` already produced, not a
second citation-extraction pass.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ai.schemas import AnswerWithCitations, Citation
from researcher.concurrency.models import SOURCE_NAMES, SourceName
from researcher.exceptions import ValidationError

# Short, informal names a person might type that map onto a canonical
# SOURCE_NAMES entry. Canonical names are always accepted as-is even though
# they're not listed as keys here (see `normalize_source_name`).
_ALIASES: dict[str, SourceName] = {
    "wiki": "wikipedia",
    "wikipedia": "wikipedia",
    "arxiv": "arxiv",
    "web": "web",
    "internet": "web",
    "search": "web",
}


def normalize_source_name(raw: str) -> SourceName:
    """Resolve one user-typed source name (possibly an alias) to a
    canonical `SourceName`. Raises `ValidationError` for anything
    unrecognized -- never silently drops or guesses at an unknown name.
    """
    key = raw.strip().lower()
    canonical = _ALIASES.get(key)
    if canonical is None:
        raise ValidationError(
            f"Unknown source {raw!r}. Expected one of: "
            f"{', '.join(SOURCE_NAMES)} (aliases: {', '.join(sorted(_ALIASES))})."
        )
    return canonical


def select_sources(requested: Sequence[str] | None) -> list[SourceName]:
    """Resolve a `--sources` selection into a canonical, deduplicated,
    stably-ordered list of source names.

    - `None` or an empty sequence means "use every known source" --
      returns all of `SOURCE_NAMES`, in `SOURCE_NAMES`'s own order.
    - Aliases (e.g. "wiki") are resolved via `normalize_source_name`.
    - Duplicates (including a canonical name and its alias both given,
      e.g. `["wiki", "wikipedia"]`) collapse to one entry.
    - Output order always follows `SOURCE_NAMES`'s canonical order,
      regardless of the order given -- so downstream code (numbering,
      fan-out) never depends on user input order.
    - An unrecognized name raises `ValidationError` (via
      `normalize_source_name`) -- the whole selection is rejected rather
      than silently dropping the bad entry.
    """
    if not requested:
        return list(SOURCE_NAMES)

    resolved: set[SourceName] = {normalize_source_name(name) for name in requested}
    return [name for name in SOURCE_NAMES if name in resolved]


def used_references(answer: AnswerWithCitations) -> list[Citation]:
    """Return `answer`'s citations in ascending `[N]` order.

    `ai.synthesizer.synthesize` already restricts `citations` to indices
    actually cited in `answer.answer`'s text (not every candidate source),
    so this does not filter anything out -- it only guarantees a stable,
    ascending display order regardless of what order `synthesize` happened
    to build the list in.
    """
    return sorted(answer.citations, key=lambda c: c.index)


def format_references(
    answer: AnswerWithCitations,
    *,
    citations: Iterable[Citation] | None = None,
) -> str:
    """Render a plain-text, numbered reference list for CLI display, e.g.:

        [1] Python (programming language) -- https://en.wikipedia.org/...
        [3] Some arXiv Paper -- https://arxiv.org/abs/...

    Pass `citations` explicitly to render a subset (already computed by
    `used_references` or filtered some other way); defaults to
    `used_references(answer)`. Returns an empty string when there are no
    citations to show, rather than an empty-looking bracketed line.
    """
    rows = list(citations) if citations is not None else used_references(answer)
    if not rows:
        return ""
    return "\n".join(f"[{c.index}] {c.source.title} -- {c.source.url}" for c in rows)


__all__ = [
    "format_references",
    "normalize_source_name",
    "select_sources",
    "used_references",
]