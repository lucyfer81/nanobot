import json
from datetime import datetime
from pathlib import Path

from nanobot.session.manager import SessionManager
from nanobot.session.repair import repair_session_file, scan_sessions_for_corruption


def _make_manager(tmp_path: Path, auto_repair: bool = False) -> SessionManager:
    return SessionManager(workspace=tmp_path, sessions_dir=tmp_path, auto_repair=auto_repair)


def test_scan_and_repair_corrupted_session_file(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path, auto_repair=False)
    key = "test:repair-file"
    path = manager._get_session_path(key)

    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "_type": "metadata",
                        "created_at": "2026-01-01T00:00:00",
                        "updated_at": "2026-01-01T00:00:00",
                        "metadata": {},
                        "last_consolidated": 0,
                    }
                ),
                json.dumps({"role": "user", "content": "ok", "timestamp": "2026-01-01T00:01:00"}),
                '{"role":"assistant","content":"broken"',  # broken JSON line
                json.dumps({"role": "assistant", "content": "ok2", "timestamp": "2026-01-01T00:02:00"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    issues = scan_sessions_for_corruption(tmp_path)
    assert len(issues) == 1
    assert issues[0].path == path
    assert issues[0].bad_lines == 1

    result = repair_session_file(path)
    assert result.repaired is True
    assert result.bad_lines == 1
    assert result.quarantine_path is not None
    assert result.quarantine_path.exists()

    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = json.loads(line)
        assert isinstance(parsed, dict)

    reloaded = _make_manager(tmp_path, auto_repair=False).get_or_create(key)
    assert len(reloaded.messages) == 2


def test_load_tolerates_bad_lines_and_missing_metadata(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path, auto_repair=False)
    key = "test:missing-meta"
    path = manager._get_session_path(key)

    path.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "hello", "timestamp": "2026-01-02T00:00:00"}),
                '{"role":"assistant","content":"bad"',  # broken JSON line
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    session = manager.get_or_create(key)
    assert len(session.messages) == 1
    assert session.metadata == {}
    assert isinstance(session.created_at, datetime)
    assert isinstance(session.updated_at, datetime)


def test_startup_best_effort_repair_hook(tmp_path: Path) -> None:
    seed_manager = _make_manager(tmp_path, auto_repair=False)
    key = "test:startup-repair"
    path = seed_manager._get_session_path(key)
    path.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "before-meta"}),
                json.dumps(
                    {
                        "_type": "metadata",
                        "created_at": "2026-01-03T00:00:00",
                        "updated_at": "2026-01-03T00:00:00",
                        "metadata": {},
                        "last_consolidated": 0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Instantiating manager triggers startup scan + best-effort repair.
    _ = _make_manager(tmp_path, auto_repair=True)

    issues_after = scan_sessions_for_corruption(tmp_path)
    assert issues_after == []
