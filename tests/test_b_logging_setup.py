"""Tests for researcher/services/logging_setup.py (B2)."""

from __future__ import annotations

import io
import logging

from researcher.services import logging_setup


def _install_capture_handler(root: logging.Logger) -> io.StringIO:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging_setup.ExtraFieldsFormatter("%(levelname)s %(message)s"))
    root.addHandler(handler)
    return buf


class TestExtraFieldsFormatter:
    def test_renders_extra_fields(self):
        logging_setup.reset_for_tests()
        logging_setup.configure_logging(level="INFO", force=True)
        root = logging.getLogger()
        buf = _install_capture_handler(root)

        log = logging.getLogger("test.b.logging")
        log.warning("fetch_exhausted", extra={"source": "web", "error_code": "provider_error"})

        output = buf.getvalue()
        assert "fetch_exhausted" in output
        assert "source='web'" in output
        assert "error_code='provider_error'" in output

    def test_no_extras_does_not_add_trailing_separator(self):
        logging_setup.reset_for_tests()
        logging_setup.configure_logging(level="INFO", force=True)
        root = logging.getLogger()
        buf = _install_capture_handler(root)

        log = logging.getLogger("test.b.logging.plain")
        log.info("plain message, no extras")

        output = buf.getvalue()
        assert "plain message, no extras" in output
        assert "|" not in output


class TestConfigureLogging:
    def test_is_idempotent_without_force(self):
        logging_setup.reset_for_tests()
        logging_setup.configure_logging(level="INFO", force=True)
        root = logging.getLogger()
        handlers_before = len(root.handlers)

        logging_setup.configure_logging(level="INFO")  # no force -> no-op

        assert len(root.handlers) == handlers_before

    def test_force_replaces_handlers_not_duplicates(self):
        logging_setup.reset_for_tests()
        logging_setup.configure_logging(level="INFO", force=True)
        root = logging.getLogger()
        handlers_after_first = len(root.handlers)

        logging_setup.configure_logging(level="DEBUG", force=True)

        assert len(root.handlers) == handlers_after_first
        assert root.level == logging.DEBUG

    def test_suppresses_httpx_and_httpcore_noise(self):
        logging_setup.reset_for_tests()
        logging_setup.configure_logging(level="DEBUG", force=True)

        assert logging.getLogger("httpx").level >= logging.WARNING
        assert logging.getLogger("httpcore").level >= logging.WARNING

    def test_respects_explicit_level_over_env(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        logging_setup.reset_for_tests()
        logging_setup.configure_logging(level="DEBUG", force=True)

        assert logging.getLogger().level == logging.DEBUG

    def test_falls_back_to_env_var_when_no_level_given(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        logging_setup.reset_for_tests()
        logging_setup.configure_logging(force=True)

        assert logging.getLogger().level == logging.ERROR