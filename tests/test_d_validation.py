import pytest

from researcher.exceptions import ValidationError
from researcher.validation import validate_limit, validate_question


def test_question_normalizes_whitespace():
    result = validate_question("  What\tis\nphotosynthesis?  ")

    assert result == "What is photosynthesis?"


@pytest.mark.parametrize("length", [3, 500])
def test_question_accepts_boundary_lengths(length):
    question = "a" * length

    assert validate_question(question) == question


@pytest.mark.parametrize("question", [None, "", " ", "\t\n"])
def test_question_rejects_empty_input(question):
    with pytest.raises(ValidationError, match="cannot be empty"):
        validate_question(question)


@pytest.mark.parametrize("question", ["a", "ab", "  ab  "])
def test_question_rejects_short_input(question):
    with pytest.raises(ValidationError, match="at least 3"):
        validate_question(question)


def test_question_rejects_long_input():
    with pytest.raises(ValidationError, match="cannot exceed 500"):
        validate_question("a" * 501)


def test_question_checks_length_after_normalizing():
    question = "  " + ("a" * 500) + "\n"

    assert validate_question(question) == "a" * 500


@pytest.mark.parametrize("limit", [1, 10, 100])
def test_limit_accepts_valid_values(limit):
    assert validate_limit(limit) == limit


@pytest.mark.parametrize("limit", [-1, 0, 101])
def test_limit_rejects_out_of_range_values(limit):
    with pytest.raises(ValidationError, match="between 1 and 100"):
        validate_limit(limit)