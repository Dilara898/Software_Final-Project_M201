import pytest
from pydantic import ValidationError


@pytest.mark.parametrize("value", ["0", "-1", "11", "nan", "inf"])
def test_cache_deadline_rejects_invalid_environment(value, monkeypatch):
    from researcher.config import Settings

    monkeypatch.setenv("CACHE_TIMEOUT_SECONDS", value)
    with pytest.raises(ValidationError):
        Settings(database_url="postgresql://localhost/test", _env_file=None)


def test_cache_deadline_reads_environment(monkeypatch):
    from researcher.config import Settings

    monkeypatch.setenv("CACHE_TIMEOUT_SECONDS", "1.25")
    settings = Settings(database_url="postgresql://localhost/test", _env_file=None)
    assert settings.cache_timeout_seconds == 1.25


def test_missing_database_url_raises(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from researcher.config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_is_cached(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:x@localhost:5432/x")
    from researcher.config import get_settings
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()


def test_importing_config_module_does_not_require_database_url(monkeypatch):
    """Modulu import etmək settings yaratmır — yalnız get_settings() çağırışı yaradır."""
    import importlib
    import researcher.config as config_module
    importlib.reload(config_module)
    assert hasattr(config_module, "get_settings")
    assert not hasattr(config_module, "settings")
