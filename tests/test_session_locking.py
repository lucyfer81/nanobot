import json
import threading
import time
from pathlib import Path

from nanobot.session.manager import Session, SessionManager


def _make_manager(tmp_path: Path) -> SessionManager:
    return SessionManager(workspace=tmp_path, sessions_dir=tmp_path, auto_repair=False)


def test_save_waits_when_same_session_lock_is_held(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    session = Session(key="test:locked")
    session.add_message("user", "hello")

    done = threading.Event()

    def writer() -> None:
        manager.save(session)
        done.set()

    with manager._acquire_session_lock(session.key):
        t = threading.Thread(target=writer, daemon=True)
        t.start()
        time.sleep(0.1)
        assert not done.is_set()

    t.join(timeout=2.0)
    assert done.is_set()
    assert manager._get_lock_path(session.key).exists()


def test_concurrent_saves_keep_jsonl_consistent(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    key = "test:concurrent"

    def writer(worker_id: int) -> None:
        for round_id in range(40):
            session = Session(key=key)
            session.add_message("user", f"worker-{worker_id}-{round_id}")
            manager.save(session)

    threads = [threading.Thread(target=writer, args=(idx,), daemon=True) for idx in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive()

    path = manager._get_session_path(key)
    assert path.exists()
    for line in path.read_text(encoding="utf-8").splitlines():
        assert isinstance(json.loads(line), dict)
