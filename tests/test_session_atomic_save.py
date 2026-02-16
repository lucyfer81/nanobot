import json
from pathlib import Path

import pytest

from nanobot.session.manager import Session, SessionManager


def _make_manager(tmp_path: Path) -> SessionManager:
    return SessionManager(workspace=tmp_path, sessions_dir=tmp_path, auto_repair=False)


def test_save_writes_valid_jsonl_file(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    session = Session(key="test:atomic-valid")
    session.add_message("user", "hello")
    session.add_message("assistant", "world")

    manager.save(session)

    path = manager._get_session_path(session.key)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        parsed = json.loads(line)
        assert isinstance(parsed, dict)

    first = json.loads(lines[0])
    assert first.get("_type") == "metadata"


def test_save_keeps_previous_file_when_rename_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _make_manager(tmp_path)
    session = Session(key="test:atomic-rollback")
    session.add_message("user", "first")
    manager.save(session)

    path = manager._get_session_path(session.key)
    original_content = path.read_text(encoding="utf-8")

    session.add_message("assistant", "second")

    import nanobot.session.manager as manager_module

    def fail_replace(_src: str, _dst: str) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(manager_module.os, "replace", fail_replace)
    with pytest.raises(OSError):
        manager.save(session)

    assert path.read_text(encoding="utf-8") == original_content
