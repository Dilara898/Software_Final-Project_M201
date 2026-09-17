import pytest
from pydantic import ValidationError


def test_missing_database_url_raises(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from researcher.config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=None)