"""Validates CLI input: question text and history limit."""

from researcher.exceptions import ValidationError

MIN_QUESTION_LENGTH = 3
MAX_QUESTION_LENGTH = 500

MIN_LIMIT = 1
MAX_LIMIT = 100


def validate_question(raw: str) -> str:
    """Cleans up whitespace and rejects bad-length questions."""
    text = " ".join((raw or "").split())

    if not text:
        raise ValidationError("Question cannot be empty.")

    if len(text) < MIN_QUESTION_LENGTH:
        raise ValidationError(
            f"Question must be at least {MIN_QUESTION_LENGTH} characters long."
        )

    if len(text) > MAX_QUESTION_LENGTH:
        raise ValidationError(
            f"Question cannot exceed {MAX_QUESTION_LENGTH} characters "
            f"(yours is {len(text)})."
        )

    return text


def validate_limit(value: int) -> int:
    """Rejects --limit values outside 1-100."""
    if not MIN_LIMIT <= value <= MAX_LIMIT:
        raise ValidationError(
            f"--limit must be between {MIN_LIMIT} and {MAX_LIMIT} "
            f"(yours is {value})."
        )
    return value