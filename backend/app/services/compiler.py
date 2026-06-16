from pathlib import Path

from app.config import get_settings
from app.database import db_session, utc_now
from app.services.outputs import slugify


def compile_source_summaries(limit: int = 25) -> dict:
    """Create simple source summary pages for recently ingested files."""
    settings = get_settings()
    output_dir = settings.resolved_wiki_dir() / "sources"
    output_dir.mkdir(parents=True, exist_ok=True)
    created = 0

    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT rf.id, rf.relative_path, rf.source_type,
                   ed.title, ed.text_preview, ed.word_count, ed.extraction_method
            FROM raw_files rf
            JOIN extracted_documents ed ON ed.file_id = rf.id
            WHERE rf.status='processed'
            ORDER BY rf.processed_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    valid_slugs = {slugify(row["relative_path"]) for row in rows}
    for page in output_dir.glob("*.md"):
        if page.stem not in valid_slugs:
            page.unlink()

    for row in rows:
        slug = slugify(row["relative_path"])
        path = output_dir / f"{slug}.md"
        body = [
            "---",
            f"title: {row['title']}",
            f"slug: {slug}",
            "type: source_summary",
            "status: draft",
            f"created: {utc_now()}",
            f"updated: {utc_now()}",
            f"source_path: {row['relative_path']}",
            "---",
            "",
            f"# {row['title']}",
            "",
            "## Summary",
            "",
            "This source was ingested by Cognix and is ready for deeper compilation.",
            "",
            "## Extraction",
            "",
            f"- Source type: {row['source_type']}",
            f"- Extraction method: {row['extraction_method']}",
            f"- Word count: {row['word_count']}",
            "",
            "## Preview",
            "",
            row["text_preview"],
        ]
        path.write_text("\n".join(body) + "\n", encoding="utf-8")
        created += 1

    _write_source_index(output_dir)
    return {"created": created}


def _write_source_index(source_dir: Path) -> None:
    index_path = get_settings().resolved_wiki_dir() / "_indexes" / "source-index.md"
    pages = sorted(path for path in source_dir.glob("*.md"))
    lines = ["# Source Index", "", "Generated source summaries.", ""]
    for page in pages:
        lines.append(f"- [[{page.stem}]]")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
