"""Lite memory retrieval: SQLite FTS5 first, substring fallback second."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
import sqlite3
from pathlib import Path
from typing import Any
import warnings

from loguru import logger

from nanobot.utils.helpers import ensure_dir

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"jieba\._compat",
)

try:
    import rjieba  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency fallback
    rjieba = None  # type: ignore[assignment]

try:
    import jieba  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency fallback
    jieba = None  # type: ignore[assignment]

ENTRY_HEADER_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
CJK_BLOCK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


@dataclass
class ParsedMemoryEntry:
    """Normalized memory entry used for indexing and retrieval."""

    file: str
    title: str
    created_at: str
    entry_type: str
    tags: str
    tldr: str
    details: str
    full_text: str


class MemoryRetriever:
    """Built-in retriever: SQLite FTS5 first, substring fallback second."""

    def __init__(
        self,
        workspace: Path,
        *,
        backend: str = "builtin_fts",
        force_disable_fts: bool = False,
    ) -> None:
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.db = self.memory_dir / "index.db"
        self.backend = backend
        self.force_disable_fts = force_disable_fts
        self._fts_available = False
        self._source_signature: tuple[tuple[str, int, int], ...] | None = None
        self._sync_index_if_needed(force=True)

    def refresh_index(self) -> None:
        """Force a full index refresh."""
        self._sync_index_if_needed(force=True)

    def probe(self, query: str) -> float:
        """
        Run an ultra-light probe and return top-1 score.

        Probe is used by OFF tier routing to detect if a retrieval upgrade may help.
        """
        if not query.strip():
            return 0.0
        rows = self.search_light(query, top_k=1)
        if not rows:
            return 0.0
        return float(rows[0].get("score") or 0.0)

    def search_light(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Tier-1 search: prioritize `title + tags + tl;dr`.

        Returns rows with source metadata for optional prompt injection.
        """
        if not query.strip():
            return []
        self._sync_index_if_needed()
        limit = max(1, top_k)
        if self._use_fts():
            rows = self._search_fts(query, limit=limit, heavy=False)
            if rows:
                return rows
        return self._search_substring(query, limit=limit, heavy=False)

    def search_heavy(
        self,
        query: str,
        top_k: int = 8,
        *,
        max_snippet_chars: int = 360,
    ) -> list[dict[str, Any]]:
        """
        Tier-2 search: full-text retrieval with ranking, dedupe and snippet compression.
        """
        if not query.strip():
            return []
        self._sync_index_if_needed()
        limit = max(1, top_k)
        candidates = limit * 3
        if self._use_fts():
            rows = self._search_fts(
                query,
                limit=candidates,
                heavy=True,
                max_snippet_chars=max_snippet_chars,
            )
            if rows:
                return self._dedupe_ranked(rows, top_k=limit)
        rows = self._search_substring(
            query,
            limit=candidates,
            heavy=True,
            max_snippet_chars=max_snippet_chars,
        )
        return self._dedupe_ranked(rows, top_k=limit)

    def _sync_index_if_needed(self, force: bool = False) -> None:
        """Refresh index when source files changed."""
        signature = self._build_source_signature()
        if not force and signature == self._source_signature:
            return
        with self._connect() as conn:
            self._ensure_schema(conn)
            self._rebuild_index(conn)
        self._source_signature = signature

    def _use_fts(self) -> bool:
        return self.backend != "grep_only" and self._fts_available and not self.force_disable_fts

    def _connect(self) -> sqlite3.Connection:
        ensure_dir(self.memory_dir)
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file TEXT NOT NULL,
                title TEXT,
                created_at TEXT,
                entry_type TEXT,
                tags TEXT,
                tldr TEXT,
                details TEXT,
                full_text TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_created_at ON entries(created_at)")

        self._fts_available = False
        if self.backend == "grep_only" or self.force_disable_fts:
            return

        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
                USING fts5(title, tags, tldr, details, full_text)
                """
            )
            self._fts_available = True
        except sqlite3.OperationalError as exc:
            self._fts_available = False
            logger.warning("Memory retrieval: FTS5 unavailable, fallback to substring search: {}", exc)

    def _rebuild_index(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM entries")
        if self._use_fts():
            conn.execute("DELETE FROM entries_fts")

        for file_path in self._iter_source_files():
            for entry in self._parse_file_entries(file_path):
                cursor = conn.execute(
                    """
                    INSERT INTO entries(file, title, created_at, entry_type, tags, tldr, details, full_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.file,
                        entry.title,
                        entry.created_at,
                        entry.entry_type,
                        entry.tags,
                        entry.tldr,
                        entry.details,
                        entry.full_text,
                    ),
                )
                if self._use_fts():
                    conn.execute(
                        """
                        INSERT INTO entries_fts(rowid, title, tags, tldr, details, full_text)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            cursor.lastrowid,
                            self._segment_for_index(entry.title),
                            self._segment_for_index(entry.tags),
                            self._segment_for_index(entry.tldr),
                            self._segment_for_index(entry.details),
                            self._segment_for_index(entry.full_text),
                        ),
                    )
        conn.commit()

    def _iter_source_files(self) -> list[Path]:
        files = []
        for path in sorted(self.memory_dir.glob("*.md")):
            if path.name.lower() == "history.md":
                continue
            files.append(path)
        return files

    def _build_source_signature(self) -> tuple[tuple[str, int, int], ...]:
        rows: list[tuple[str, int, int]] = []
        for path in self._iter_source_files():
            try:
                stat = path.stat()
            except OSError:
                continue
            rows.append((path.name, int(stat.st_mtime), stat.st_size))
        return tuple(rows)

    def _parse_file_entries(self, file_path: Path) -> list[ParsedMemoryEntry]:
        try:
            raw = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.debug("Memory retrieval: failed reading {}: {}", file_path, exc)
            return []

        matches = list(ENTRY_HEADER_RE.finditer(raw))
        if not matches:
            return self._parse_legacy_file(file_path, raw)

        entries: list[ParsedMemoryEntry] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
            block = raw[start:end].strip()
            title = match.group(1).strip()
            entry = self._parse_structured_block(file_path, title, block)
            entries.append(entry)
        return entries

    def _parse_legacy_file(self, file_path: Path, raw: str) -> list[ParsedMemoryEntry]:
        text = raw.strip()
        if not text:
            return []
        created = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        return [
            ParsedMemoryEntry(
                file=self._relative_file(file_path),
                title=file_path.stem,
                created_at=created,
                entry_type="note",
                tags="",
                tldr=self._guess_tldr(text),
                details=text,
                full_text=text,
            )
        ]

    def _parse_structured_block(self, file_path: Path, title: str, block: str) -> ParsedMemoryEntry:
        lines = block.splitlines()
        body = lines[1:] if len(lines) > 1 else []
        entry_type = "note"
        tags = ""
        tldr = ""
        details_lines: list[str] = []
        details_mode = False
        free_lines: list[str] = []

        for raw_line in body:
            line = raw_line.rstrip()
            stripped = line.strip()
            lower = stripped.lower()
            if lower.startswith("type:"):
                entry_type = stripped.split(":", 1)[1].strip() or "note"
                continue
            if lower.startswith("tags:"):
                tags = self._normalize_tags(stripped.split(":", 1)[1].strip())
                continue
            if lower.startswith("tl;dr:"):
                tldr = stripped.split(":", 1)[1].strip()
                continue
            if lower.startswith("details:"):
                details_mode = True
                after = stripped.split(":", 1)[1].strip()
                if after:
                    details_lines.append(after)
                continue

            if details_mode:
                details_lines.append(line)
            elif stripped:
                free_lines.append(stripped)

        details = "\n".join([line for line in details_lines if line.strip()]).strip()
        if not details:
            details = "\n".join(free_lines).strip()

        if not tldr:
            tldr = self._guess_tldr(details or " ".join(free_lines) or title)

        created = self._normalize_timestamp(title, fallback_path=file_path)
        normalized_tags = self._normalize_tags(tags)
        clean_type = (entry_type or "note").strip().lower()
        if clean_type not in {"decision", "note", "bug", "idea", "config"}:
            clean_type = "note"

        full_text = "\n".join(body).strip() or f"tl;dr: {tldr}\n{details}".strip()
        return ParsedMemoryEntry(
            file=self._relative_file(file_path),
            title=title,
            created_at=created,
            entry_type=clean_type,
            tags=normalized_tags,
            tldr=tldr,
            details=details,
            full_text=full_text,
        )

    def _search_fts(
        self,
        query: str,
        *,
        limit: int,
        heavy: bool,
        max_snippet_chars: int = 360,
    ) -> list[dict[str, Any]]:
        match_query = self._build_fts_query(query)
        if not match_query:
            return []

        weight_expr = (
            "bm25(entries_fts, 0.8, 1.0, 1.3, 1.6, 2.0)"
            if heavy
            else "bm25(entries_fts, 2.2, 1.8, 2.4, 0.3, 0.1)"
        )
        sql = f"""
            SELECT
                e.file,
                e.title,
                e.created_at,
                e.entry_type,
                e.tags,
                e.tldr,
                e.details,
                e.full_text,
                {weight_expr} AS lexical_score
            FROM entries_fts
            JOIN entries e ON e.id = entries_fts.rowid
            WHERE entries_fts MATCH ?
            ORDER BY lexical_score ASC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (match_query, limit)).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            lexical = self._lexical_from_bm25(float(row["lexical_score"]))
            recency = self._recency_boost(str(row["created_at"] or ""))
            final_score = lexical * 0.75 + recency * 0.25
            snippet_source = str(row["full_text"] or row["details"] or row["tldr"] or "")
            snippet = self._build_snippet(
                snippet_source,
                query=query,
                max_chars=max_snippet_chars,
            )
            results.append(
                self._format_row(
                    row,
                    final_score=final_score,
                    snippet=snippet,
                    include_snippet=heavy,
                )
            )

        results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return results

    def _search_substring(
        self,
        query: str,
        *,
        limit: int,
        heavy: bool,
        max_snippet_chars: int = 360,
    ) -> list[dict[str, Any]]:
        terms = self._query_terms(query)
        if not terms:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT file, title, created_at, entry_type, tags, tldr, details, full_text
                FROM entries
                """
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            light_text = " ".join([str(row["title"] or ""), str(row["tags"] or ""), str(row["tldr"] or "")])
            heavy_text = " ".join(
                [
                    light_text,
                    str(row["details"] or ""),
                    str(row["full_text"] or ""),
                ]
            )
            searchable = heavy_text if heavy else light_text
            lower = searchable.lower()
            hit_count = sum(lower.count(term.lower()) for term in terms if term)
            if hit_count <= 0:
                continue
            lexical = min(1.0, hit_count / max(1.0, float(len(terms) * 2)))
            recency = self._recency_boost(str(row["created_at"] or ""))
            final_score = lexical * 0.75 + recency * 0.25
            snippet = self._build_snippet(heavy_text, query=query, max_chars=max_snippet_chars)
            results.append(
                self._format_row(
                    row,
                    final_score=final_score,
                    snippet=snippet,
                    include_snippet=heavy,
                )
            )

        results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return results[:limit]

    def _dedupe_ranked(self, rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
        dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            key = (
                str(row.get("file") or ""),
                str(row.get("date") or ""),
                str(row.get("tldr") or ""),
            )
            prev = dedup.get(key)
            if prev is None or float(row.get("score") or 0.0) > float(prev.get("score") or 0.0):
                dedup[key] = row
        ranked = sorted(dedup.values(), key=lambda item: item.get("score", 0.0), reverse=True)
        return ranked[: max(1, top_k)]

    def _format_row(
        self,
        row: sqlite3.Row | dict[str, Any],
        *,
        final_score: float,
        snippet: str,
        include_snippet: bool,
    ) -> dict[str, Any]:
        tags_value = str(row["tags"] or "")
        item: dict[str, Any] = {
            "file": str(row["file"]),
            "title": str(row["title"] or ""),
            "date": str(row["created_at"] or ""),
            "type": str(row["entry_type"] or "note"),
            "tags": [tag for tag in [t.strip() for t in tags_value.split(",")] if tag],
            "tldr": str(row["tldr"] or ""),
            "score": round(max(0.0, float(final_score)), 6),
        }
        if include_snippet:
            item["snippet"] = snippet
        return item

    def _build_fts_query(self, query: str) -> str:
        tokens = self._query_terms(query, max_terms=16)
        if not tokens:
            return ""
        selected = [token.replace('"', "") for token in tokens[:16]]
        return " OR ".join(f'"{token}"' for token in selected if token)

    @staticmethod
    def _query_terms(query: str, *, max_terms: int = 16) -> list[str]:
        text = (query or "").strip()
        if not text:
            return []

        tokens: list[str] = []

        # Keep latin/numeric tokens as-is (lowercased), useful for commands/ports.
        for word in TOKEN_RE.findall(text):
            normalized = word.strip().lower()
            if len(normalized) >= 2:
                tokens.append(normalized)

        # Segment contiguous CJK blocks with rjieba/jieba; fallback to 2-gram.
        for block in CJK_BLOCK_RE.findall(text):
            block = block.strip()
            if not block:
                continue
            tokens.extend(MemoryRetriever._segment_cjk(block))

        # Fallback for punctuation-separated text when regex extraction missed content.
        if not tokens:
            rough = [token.strip() for token in re.split(r"[\s,，。！？;；]+", text) if token.strip()]
            tokens.extend(token.lower() for token in rough)

        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            normalized = token.strip().lower()
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
            if len(deduped) >= max(1, max_terms):
                break
        return deduped

    @staticmethod
    def _segment_cjk(text: str) -> list[str]:
        clean = "".join(ch for ch in text.strip() if CJK_CHAR_RE.match(ch))
        if not clean:
            return []

        if rjieba is not None:
            try:
                pieces = [
                    token.strip()
                    for token in rjieba.cut(clean, False)  # type: ignore[union-attr]
                    if token and token.strip()
                ]
                # Drop noisy single-char tokens; keep meaningful multi-char words.
                segmented = [token for token in pieces if len(token) >= 2]
                if segmented:
                    return segmented
            except Exception:
                pass

        if jieba is not None:
            try:
                pieces = [
                    token.strip()
                    for token in jieba.lcut(clean, HMM=False)  # type: ignore[union-attr]
                    if token and token.strip()
                ]
                # Drop noisy single-char tokens; keep meaningful multi-char words.
                segmented = [token for token in pieces if len(token) >= 2]
                if segmented:
                    return segmented
            except Exception:
                pass

        # Fallback: 2-gram segmentation for CJK when segmenters are unavailable.
        if len(clean) <= 2:
            return [clean]
        return [clean[idx : idx + 2] for idx in range(0, len(clean) - 1)]

    def _segment_for_index(self, text: str) -> str:
        tokens = self._query_terms(text, max_terms=512)
        return " ".join(tokens)

    @staticmethod
    def _lexical_from_bm25(score: float) -> float:
        if score < 0:
            return 1.0
        return 1.0 / (1.0 + score)

    @staticmethod
    def _recency_boost(created_at: str) -> float:
        if not created_at:
            return 0.1
        parsed = MemoryRetriever._parse_datetime(created_at)
        if parsed is None:
            return 0.1
        age_days = max(0.0, (datetime.now() - parsed).total_seconds() / 86_400.0)
        return 1.0 / (1.0 + age_days / 30.0)

    @staticmethod
    def _build_snippet(text: str, *, query: str, max_chars: int) -> str:
        compact = " ".join((text or "").split())
        if not compact:
            return ""
        budget = max(60, max_chars)
        if len(compact) <= budget:
            return compact

        terms = MemoryRetriever._query_terms(query)
        lower = compact.lower()
        hit_index = -1
        for term in terms:
            idx = lower.find(term.lower())
            if idx >= 0:
                hit_index = idx
                break

        if hit_index < 0:
            snippet = compact[:budget].rstrip()
            return f"{snippet}..."

        left_span = budget // 3
        start = max(0, hit_index - left_span)
        end = min(len(compact), start + budget)
        snippet = compact[start:end].strip()
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(compact) else ""
        return f"{prefix}{snippet}{suffix}"

    @staticmethod
    def _normalize_tags(tags: str) -> str:
        if not tags:
            return ""
        parts = [part.strip() for part in tags.split(",")]
        clean = [part for part in parts if part]
        return ", ".join(clean)

    @staticmethod
    def _guess_tldr(text: str) -> str:
        compact = " ".join(text.split())
        if not compact:
            return ""
        return compact[:60] if len(compact) <= 60 else f"{compact[:57]}..."

    def _normalize_timestamp(self, title: str, *, fallback_path: Path) -> str:
        parsed = self._parse_datetime(title)
        if parsed is not None:
            return parsed.strftime("%Y-%m-%d %H:%M")
        fallback = datetime.fromtimestamp(fallback_path.stat().st_mtime)
        return fallback.strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        text = value.strip()
        if not text:
            return None
        # Keep only the leading datetime part when title has suffix text.
        text = text.split("  ", 1)[0].strip()
        for fmt in (
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d",
        ):
            try:
                parsed = datetime.strptime(text, fmt)
                if fmt in {"%Y-%m-%d", "%Y/%m/%d"}:
                    parsed = parsed.replace(hour=0, minute=0)
                return parsed
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    def _relative_file(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace))
        except ValueError:
            return str(path)
