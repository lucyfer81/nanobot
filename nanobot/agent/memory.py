"""Memory system for persistent agent memory."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger

from nanobot.utils.helpers import ensure_dir

ALLOWED_MEMORY_TYPES = {"decision", "note", "bug", "idea", "config"}


def normalize_memory_type(value: str | None) -> str:
    """Normalize memory type into allowed values."""
    normalized = (value or "note").strip().lower()
    return normalized if normalized in ALLOWED_MEMORY_TYPES else "note"


def normalize_tags(tags: list[str] | tuple[str, ...] | set[str] | str | None) -> list[str]:
    """Normalize tags from list or comma-separated text."""
    if tags is None:
        return []
    if isinstance(tags, str):
        raw = tags.split(",")
    else:
        raw = list(tags)
    clean: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            clean.append(text)
    return clean


def guess_tldr(text: str, *, max_chars: int = 60) -> str:
    """Generate a compact tl;dr from free-form text."""
    compact = " ".join((text or "").split())
    if not compact:
        return ""
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


def build_memory_entry(
    *,
    memory_type: str = "note",
    tags: list[str] | tuple[str, ...] | set[str] | str | None = None,
    tldr: str = "",
    details: str = "",
    timestamp: datetime | None = None,
) -> str:
    """Build one structured memory entry in Markdown."""
    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M")
    normalized_type = normalize_memory_type(memory_type)
    tag_line = ", ".join(normalize_tags(tags))
    details_text = (details or "").strip()
    tldr_text = " ".join((tldr or "").split()) or guess_tldr(details_text)
    if not details_text:
        details_text = tldr_text

    lines = [
        f"### {ts}",
        f"type: {normalized_type}",
        f"tags: {tag_line}",
        f"tl;dr: {tldr_text}",
        "",
        "details:",
    ]
    lines.extend(details_text.splitlines() or ["- (empty)"])
    return "\n".join(lines).rstrip()


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path, *, auto_refresh_index: bool = True):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.auto_refresh_index = auto_refresh_index

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")
        self.refresh_retrieval_index()

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def append_long_term_entry(
        self,
        entry: str,
        *,
        memory_type: str = "note",
        tags: list[str] | tuple[str, ...] | set[str] | str | None = None,
        tldr: str | None = None,
    ) -> None:
        """Append one durable entry into MEMORY.md without rewriting existing content."""
        block = self._normalize_entry_block(
            entry,
            memory_type=memory_type,
            tags=tags,
            tldr=tldr,
        )
        if not block:
            return

        if not self.memory_file.exists():
            self.memory_file.write_text("# Long-term Memory\n\n", encoding="utf-8")

        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{block}\n")
        self.refresh_retrieval_index()

    def append_long_term_structured(
        self,
        *,
        memory_type: str,
        tags: list[str] | tuple[str, ...] | set[str] | str | None,
        tldr: str,
        details: str,
    ) -> None:
        """Append one structured durable memory entry."""
        block = build_memory_entry(
            memory_type=memory_type,
            tags=tags,
            tldr=tldr,
            details=details,
        )
        self.append_long_term_entry(block)

    def append_daily_note(
        self,
        day: str,
        note: str,
        *,
        memory_type: str = "note",
        tags: list[str] | tuple[str, ...] | set[str] | str | None = None,
        tldr: str | None = None,
    ) -> Path:
        """Append one structured note into memory/YYYY-MM-DD.md."""
        note_file = self.memory_dir / f"{day}.md"
        block = self._normalize_entry_block(
            note,
            memory_type=memory_type,
            tags=tags,
            tldr=tldr,
        )
        if not block:
            return note_file

        if not note_file.exists():
            note_file.write_text(f"# Daily Notes {day}\n", encoding="utf-8")

        with open(note_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{block}\n")
        self.refresh_retrieval_index()
        return note_file

    def append_daily_structured(
        self,
        *,
        day: str,
        memory_type: str,
        tags: list[str] | tuple[str, ...] | set[str] | str | None,
        tldr: str,
        details: str,
    ) -> Path:
        """Append one structured daily note."""
        block = build_memory_entry(
            memory_type=memory_type,
            tags=tags,
            tldr=tldr,
            details=details,
        )
        return self.append_daily_note(day, block)

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    def refresh_retrieval_index(self) -> None:
        """Refresh SQLite memory index (best effort)."""
        if not self.auto_refresh_index:
            return
        try:
            from nanobot.agent.memory_retrieval import MemoryRetriever

            MemoryRetriever(
                self.workspace,
                backend="builtin_fts",
            ).refresh_index()
        except Exception as exc:
            logger.debug("Memory index refresh skipped: {}", exc)

    @staticmethod
    def _looks_structured_entry(text: str) -> bool:
        stripped = text.strip()
        if not stripped.startswith("### "):
            return False
        lowered = stripped.lower()
        return "type:" in lowered and "tl;dr:" in lowered

    def _normalize_entry_block(
        self,
        entry: str,
        *,
        memory_type: str,
        tags: list[str] | tuple[str, ...] | set[str] | str | None,
        tldr: str | None,
    ) -> str:
        text = (entry or "").strip()
        if not text:
            return ""
        if self._looks_structured_entry(text):
            return text
        return build_memory_entry(
            memory_type=memory_type,
            tags=tags,
            tldr=tldr or guess_tldr(text),
            details=text,
        )
