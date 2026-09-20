"""Covers researcher/__main__.py's `if __name__ == "__main__"` guard, which
a plain `import researcher.__main__` never executes."""
import runpy

import pytest


def test_main_module_exits_with_cli_main_return_code(monkeypatch):
    monkeypatch.setattr("researcher.cli.main", lambda: 3)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("researcher.__main__", run_name="__main__")

    assert exc.value.code == 3
