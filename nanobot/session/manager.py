"""Session management for conversation history."""

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from loguru import logger

from nanobot.utils.helpers import ensure_dir, safe_filename

try:  # pragma: no cover - Windows fallback
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(
        self,
        max_messages: int = 500,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent messages in LLM format, optionally token-pruned."""
        history = [{"role": m["role"], "content": m["content"]} for m in self.messages[-max_messages:]]
        if max_tokens is None or max_tokens <= 0:
            return history

        from nanobot.agent.context_guard import prune_messages_by_tokens

        pruned, _ = prune_messages_by_tokens(
            history,
            budget_tokens=max_tokens,
            min_messages=2,
            prioritize_tool_messages=True,
        )
        return pruned

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    def __init__(
        self,
        workspace: Path,
        sessions_dir: Path | None = None,
        auto_repair: bool = True,
    ):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(sessions_dir or (Path.home() / ".nanobot" / "sessions"))
        self._cache: dict[str, Session] = {}
        self._auto_repair = auto_repair
        self._session_locks: dict[str, threading.Lock] = {}
        self._session_locks_guard = threading.Lock()
        self._run_startup_repair()

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _get_lock_path(self, key: str) -> Path:
        """Get lock file path for a session."""
        session_path = self._get_session_path(key)
        return session_path.with_suffix(session_path.suffix + ".lock")

    def _get_thread_lock(self, key: str) -> threading.Lock:
        """Get in-process lock for one session."""
        with self._session_locks_guard:
            if key not in self._session_locks:
                self._session_locks[key] = threading.Lock()
            return self._session_locks[key]

    @contextmanager
    def _acquire_session_lock(self, key: str) -> Iterator[None]:
        """Acquire in-process + file lock for one session."""
        thread_lock = self._get_thread_lock(key)
        thread_lock.acquire()
        lock_file = None
        try:
            lock_path = self._get_lock_path(key)
            lock_file = open(lock_path, "a+", encoding="utf-8")
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if lock_file is not None:
                try:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
                lock_file.close()
            thread_lock.release()

    def _run_startup_repair(self) -> None:
        """Best-effort corruption scan and repair on startup."""
        try:
            from nanobot.session.repair import repair_session_file, scan_sessions_for_corruption

            issues = scan_sessions_for_corruption(self.sessions_dir)
            if not issues:
                return

            logger.warning(
                "Session scan detected {} potentially corrupted files in {}",
                len(issues),
                self.sessions_dir,
            )
            for issue in issues:
                logger.warning(
                    "Session corruption: file={}, bad_lines={}, reason={}",
                    issue.path,
                    issue.bad_lines,
                    issue.reason,
                )

            if not self._auto_repair:
                return

            repaired_count = 0
            for issue in issues:
                try:
                    result = repair_session_file(issue.path)
                except Exception as exc:
                    logger.warning("Session repair failed for {}: {}", issue.path, exc)
                    continue
                if result.repaired:
                    repaired_count += 1
                    logger.warning(
                        "Session repaired: file={}, bad_lines={}, quarantine={}",
                        result.path,
                        result.bad_lines,
                        result.quarantine_path,
                    )
            if repaired_count:
                logger.info("Session startup repair completed: {} file(s) repaired", repaired_count)
        except Exception as exc:
            logger.warning("Session startup scan/repair skipped: {}", exc)

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        """Safely parse ISO datetime."""
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @classmethod
    def _infer_timestamp_from_messages(cls, messages: list[dict[str, Any]], newest: bool) -> datetime | None:
        """Infer created/updated timestamp from message entries."""
        candidates = reversed(messages) if newest else messages
        for msg in candidates:
            ts = cls._parse_datetime(msg.get("timestamp"))
            if ts is not None:
                return ts
        return None

    @staticmethod
    def _build_session_lines(session: Session) -> list[str]:
        """Serialize a session into JSONL lines."""
        metadata_line = {
            "_type": "metadata",
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata if isinstance(session.metadata, dict) else {},
            "last_consolidated": session.last_consolidated if session.last_consolidated >= 0 else 0,
        }
        lines = [json.dumps(metadata_line) + "\n"]
        lines.extend(json.dumps(msg) + "\n" for msg in session.messages)
        return lines

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        """Best-effort fsync for containing directory after rename."""
        dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            dir_fd = os.open(str(directory), dir_flags)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
        finally:
            os.close(dir_fd)

    def _atomic_write_lines(self, path: Path, lines: list[str]) -> None:
        """Write JSONL lines atomically via temp file + fsync + rename."""
        ensure_dir(path.parent)
        temp_fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_name, path)
            self._fsync_directory(path.parent)
        finally:
            if os.path.exists(temp_name):
                try:
                    os.remove(temp_name)
                except OSError:
                    pass

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)

        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            updated_at = None
            last_consolidated = 0
            bad_lines = 0
            metadata_seen = False

            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        bad_lines += 1
                        continue

                    if not isinstance(data, dict):
                        bad_lines += 1
                        continue

                    if data.get("_type") == "metadata":
                        if metadata_seen:
                            bad_lines += 1
                            continue
                        metadata_seen = True
                        raw_metadata = data.get("metadata")
                        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
                        created_at = self._parse_datetime(data.get("created_at"))
                        updated_at = self._parse_datetime(data.get("updated_at"))
                        raw_last = data.get("last_consolidated", 0)
                        if isinstance(raw_last, int) and raw_last >= 0:
                            last_consolidated = raw_last
                        else:
                            last_consolidated = 0
                    else:
                        messages.append(data)

            created_at = created_at or self._infer_timestamp_from_messages(messages, newest=False) or datetime.now()
            updated_at = updated_at or self._infer_timestamp_from_messages(messages, newest=True) or created_at
            last_consolidated = min(last_consolidated, len(messages))

            if bad_lines > 0:
                logger.warning(
                    "Session {} loaded with {} corrupted line(s) skipped",
                    key,
                    bad_lines,
                )

            return Session(
                key=key,
                messages=messages,
                created_at=created_at,
                updated_at=updated_at,
                metadata=metadata,
                last_consolidated=last_consolidated,
            )
        except Exception as e:
            logger.warning(f"Failed to load session {key}: {e}")
            return None

    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)
        lines = self._build_session_lines(session)

        with self._acquire_session_lock(session.key):
            self._atomic_write_lines(path, lines)

        self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                created_at = None
                updated_at = None
                with open(path, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(data, dict) and data.get("_type") == "metadata":
                            created_at = data.get("created_at")
                            updated_at = data.get("updated_at")
                            break

                sessions.append({
                    "key": path.stem.replace("_", ":"),
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "path": str(path)
                })
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
