"""Memory system for persistent agent memory."""

from datetime import datetime
from pathlib import Path

from nanobot.utils.helpers import ensure_dir


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def append_long_term_entry(self, entry: str) -> None:
        """Append one durable memory entry to MEMORY.md without overwriting existing content."""
        text = entry.strip()
        if not text:
            return

        if not self.memory_file.exists():
            self.memory_file.write_text("# Long-term Memory\n\n", encoding="utf-8")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        block = f"\n\n### {timestamp}\n{text}\n"
        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write(block)

    def append_daily_note(self, day: str, note: str) -> Path:
        """Append one note into memory/YYYY-MM-DD.md."""
        text = note.strip()
        note_file = self.memory_dir / f"{day}.md"
        if not text:
            return note_file

        if not note_file.exists():
            note_file.write_text(f"# Daily Notes {day}\n", encoding="utf-8")

        timestamp = datetime.now().strftime("%H:%M")
        block = f"\n\n## {timestamp}\n{text}\n"
        with open(note_file, "a", encoding="utf-8") as f:
            f.write(block)

        return note_file

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""
