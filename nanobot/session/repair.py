"""Session corruption scan and repair helpers."""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SessionRepairIssue:
    """Detected corruption signals for one session file."""

    path: Path
    total_lines: int
    bad_lines: int
    missing_metadata: bool
    metadata_lines: int
    metadata_is_first: bool

    @property
    def reason(self) -> str:
        """Human-readable reason summary."""
        reasons = []
        if self.bad_lines:
            reasons.append(f"bad_lines={self.bad_lines}")
        if self.missing_metadata:
            reasons.append("missing_metadata")
        if self.metadata_lines > 1:
            reasons.append(f"metadata_lines={self.metadata_lines}")
        if self.metadata_lines >= 1 and not self.metadata_is_first:
            reasons.append("metadata_not_first")
        return ",".join(reasons) or "unknown"


@dataclass(slots=True)
class SessionRepairResult:
    """Repair execution result for one session file."""

    path: Path
    repaired: bool
    total_lines: int
    kept_lines: int
    bad_lines: int
    quarantine_path: Path | None = None


@dataclass(slots=True)
class _SessionAnalysis:
    total_lines: int
    bad_entries: list[tuple[int, str, str]]
    metadata_entries: list[tuple[int, dict[str, Any], str]]
    message_lines: list[str]
    metadata_is_first: bool


def _analyze_session_file(path: Path) -> _SessionAnalysis:
    """Parse a session file and classify each non-empty line."""
    total_lines = 0
    bad_entries: list[tuple[int, str, str]] = []
    metadata_entries: list[tuple[int, dict[str, Any], str]] = []
    message_lines: list[str] = []
    metadata_is_first = False
    first_non_empty_seen = False

    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, raw_line in enumerate(f, start=1):
            total_lines += 1
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                bad_entries.append((lineno, raw_line.rstrip("\n"), "json_decode_error"))
                if not first_non_empty_seen:
                    first_non_empty_seen = True
                continue

            if not isinstance(payload, dict):
                bad_entries.append((lineno, raw_line.rstrip("\n"), "non_object_json"))
                if not first_non_empty_seen:
                    first_non_empty_seen = True
                continue

            is_metadata = payload.get("_type") == "metadata"
            if not first_non_empty_seen:
                metadata_is_first = is_metadata
                first_non_empty_seen = True

            if is_metadata:
                metadata_entries.append((lineno, payload, stripped))
            else:
                message_lines.append(stripped)

    return _SessionAnalysis(
        total_lines=total_lines,
        bad_entries=bad_entries,
        metadata_entries=metadata_entries,
        message_lines=message_lines,
        metadata_is_first=metadata_is_first,
    )


def _normalize_metadata(metadata: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
    """Normalize metadata line and report whether it changed."""
    now = datetime.now().isoformat()
    source = metadata.copy() if isinstance(metadata, dict) else {}

    created_at = source.get("created_at")
    updated_at = source.get("updated_at")
    meta = source.get("metadata")
    last_consolidated = source.get("last_consolidated")

    changed = False
    if source.get("_type") != "metadata":
        changed = True
    source["_type"] = "metadata"

    if not isinstance(created_at, str) or not created_at:
        source["created_at"] = now
        changed = True

    if not isinstance(updated_at, str) or not updated_at:
        source["updated_at"] = source["created_at"]
        changed = True

    if not isinstance(meta, dict):
        source["metadata"] = {}
        changed = True

    if not isinstance(last_consolidated, int) or last_consolidated < 0:
        source["last_consolidated"] = 0
        changed = True

    return source, changed


def _fsync_directory(directory: Path) -> None:
    """Best-effort fsync for containing directory after rename."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        dir_fd = os.open(str(directory), flags)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _atomic_write_lines(path: Path, lines: list[str]) -> None:
    """Write file atomically via temporary file and rename."""
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
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temp_name):
            try:
                os.remove(temp_name)
            except OSError:
                pass


def scan_sessions_for_corruption(sessions_dir: Path) -> list[SessionRepairIssue]:
    """Scan all session files and return corruption issues."""
    issues: list[SessionRepairIssue] = []
    if not sessions_dir.exists():
        return issues

    for path in sorted(sessions_dir.glob("*.jsonl")):
        try:
            analysis = _analyze_session_file(path)
        except OSError:
            continue

        metadata_lines = len(analysis.metadata_entries)
        missing_metadata = metadata_lines == 0
        has_issue = bool(analysis.bad_entries) or missing_metadata or metadata_lines > 1
        has_issue = has_issue or (metadata_lines >= 1 and not analysis.metadata_is_first)
        if not has_issue:
            continue

        issues.append(
            SessionRepairIssue(
                path=path,
                total_lines=analysis.total_lines,
                bad_lines=len(analysis.bad_entries),
                missing_metadata=missing_metadata,
                metadata_lines=metadata_lines,
                metadata_is_first=analysis.metadata_is_first,
            )
        )
    return issues


def repair_session_file(path: Path) -> SessionRepairResult:
    """Repair a corrupted session file in place.

    Strategy:
    - Keep parseable JSON object lines.
    - Keep only the first metadata line (or synthesize one).
    - Move bad lines to a quarantine file.
    """
    analysis = _analyze_session_file(path)
    bad_entries = list(analysis.bad_entries)

    metadata_lines = analysis.metadata_entries
    metadata_line = metadata_lines[0][1] if metadata_lines else None
    for lineno, _data, raw in metadata_lines[1:]:
        bad_entries.append((lineno, raw, "duplicate_metadata"))

    normalized_metadata, metadata_changed = _normalize_metadata(metadata_line)
    needs_repair = bool(bad_entries)
    needs_repair = needs_repair or len(metadata_lines) != 1
    needs_repair = needs_repair or not analysis.metadata_is_first
    needs_repair = needs_repair or metadata_changed

    clean_lines = [json.dumps(normalized_metadata) + "\n"]
    clean_lines.extend(line + "\n" for line in analysis.message_lines)

    if not needs_repair:
        return SessionRepairResult(
            path=path,
            repaired=False,
            total_lines=analysis.total_lines,
            kept_lines=len(clean_lines),
            bad_lines=0,
            quarantine_path=None,
        )

    quarantine_path = None
    if bad_entries:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        quarantine_path = path.with_name(f"{path.name}.quarantine.{stamp}.jsonl")
        quarantine_lines = []
        for lineno, content, reason in bad_entries:
            quarantine_lines.append(
                json.dumps(
                    {
                        "line": lineno,
                        "reason": reason,
                        "content": content,
                    }
                )
                + "\n"
            )
        _atomic_write_lines(quarantine_path, quarantine_lines)

    _atomic_write_lines(path, clean_lines)
    return SessionRepairResult(
        path=path,
        repaired=True,
        total_lines=analysis.total_lines,
        kept_lines=len(clean_lines),
        bad_lines=len(bad_entries),
        quarantine_path=quarantine_path,
    )
