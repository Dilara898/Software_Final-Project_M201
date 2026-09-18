"""Regression tests for B-01: researcher/exceptions.py must exist and
unblock researcher/validation.py's import (D's CLI input validation),
which the external review found raising ModuleNotFoundError at commit
3e52746.
"""

from __future__ import annotations

import pytest

from researcher.exceptions import NoSourcesError, ResearcherError, ValidationError


def test_validation_module_imports_successfully():
    """This is the exact import that failed before this module existed."""
    from researcher.validation import validate_limit, validate_question  # noqa: F401


def test_validation_error_is_a_researcher_error():
    assert issubclass(ValidationError, ResearcherError)


def test_no_sources_error_is_a_researcher_error():
    assert issubclass(NoSourcesError, ResearcherError)


def test_validate_question_raises_shared_validation_error():
    from researcher.validation import validate_question

    with pytest.raises(ValidationError):
        validate_question("")
    with pytest.raises(ValidationError):
        validate_question("ab")  # below MIN_QUESTION_LENGTH


def test_validate_limit_raises_shared_validation_error():
    from researcher.validation import validate_limit

    with pytest.raises(ValidationError):
        validate_limit(0)
    with pytest.raises(ValidationError):
        validate_limit(101)


def test_no_sources_error_default_message():
    exc = NoSourcesError()
    assert "no usable sources" in str(exc).lower()
    assert exc.reason is None


def test_no_sources_error_custom_reason():
    exc = NoSourcesError("all sources timed out")
    assert str(exc) == "all sources timed out"
    assert exc.reason == "all sources timed out"