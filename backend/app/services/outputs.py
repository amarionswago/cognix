import json
import re
from pathlib import Path

from app.config import get_settings
from app.database import db_session, utc_now
from app.services.retrieval import RetrievedChunk


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or "analysis"


def save_analysis(question: str, answer: str, chunks: list[RetrievedChunk], retrieval_summary: str) -> tuple[int, Path]:
    settings = get_settings()
    output_dir = settings.resolved_wiki_dir() / "outputs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    slug = slugify(question)
    filename = f"{now[:10]}-{slug}.md"
    path = output_dir / filename
    sources = [chunk.__dict__ for chunk in chunks]
    frontmatter = [
        "---",
        f"title: {question[:80].replace(':', '-')}",
        "type: analysis",
        "status: draft",
        f"created: {now}",
        "output_format: markdown",
        "sources_used:",
    ]
    for chunk in chunks:
        frontmatter.append(f"  - chunk_id: {chunk.chunk_id}")
    frontmatter.extend(["---", ""])
    path.write_text("\n".join(frontmatter) + answer, encoding="utf-8")

    with db_session() as conn:
        cursor = conn.execute(
            """
            INSERT INTO outputs
            (title, path, type, status, query, answer_preview, sources_json, retrieval_json,
             sensitivity, created_at, updated_at)
            VALUES (?, ?, 'analysis', 'draft', ?, ?, ?, ?, 'research', ?, ?)
            """,
            (
                question[:120],
                str(path),
                question,
                answer[:500],
                json.dumps(sources),
                json.dumps({"summary": retrieval_summary}),
                now,
                now,
            ),
        )
        return int(cursor.lastrowid), path


def list_outputs(limit: int = 100, include_archived: bool = False) -> list[dict]:
    with db_session() as conn:
        if include_archived:
            return conn.execute("SELECT * FROM outputs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return conn.execute(
            """
            SELECT * FROM outputs
            WHERE status NOT IN ('archived', 'deleted')
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()


def update_output_status(output_id: int, status: str) -> dict | None:
    with db_session() as conn:
        conn.execute("UPDATE outputs SET status=?, updated_at=? WHERE id=?", (status, utc_now(), output_id))
        return conn.execute("SELECT * FROM outputs WHERE id=?", (output_id,)).fetchone()
