"""Export fine-tuning-ready Cognix examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.database import db_session


def export_training_jsonl(task: str = "qa_citation", quality_label: str | None = None) -> Path:
    """Export stored training examples to a JSONL file."""
    export_dir = get_settings().resolved_data_dir() / "exports" / "training"
    export_dir.mkdir(parents=True, exist_ok=True)
    label_suffix = quality_label or "all"
    path = export_dir / f"{task}-{label_suffix}.jsonl"
    clause = "WHERE task=?"
    params: list[str] = [task]
    if quality_label:
        clause += " AND quality_label=?"
        params.append(quality_label)
    with db_session() as conn:
        rows = conn.execute(
            f"""
            SELECT task, input_json, output_json, source, quality_label, created_at
            FROM training_examples
            {clause}
            ORDER BY id
            """,
            tuple(params),
        ).fetchall()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {
                "task": row["task"],
                "input": json.loads(row["input_json"]),
                "output": json.loads(row["output_json"]),
                "source": row["source"],
                "quality_label": row["quality_label"],
                "created_at": row["created_at"],
            }
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    return path


def export_sft_jsonl(task: str = "qa_citation", quality_label: str | None = None) -> Path:
    """Export training examples in chat-style supervised fine-tuning JSONL."""
    source_path = export_training_jsonl(task, quality_label)
    output_path = source_path.with_name(source_path.stem + "-sft.jsonl")
    with source_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as output:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            messages = training_record_to_messages(record)
            if not messages:
                continue
            output.write(json.dumps({"messages": messages}, ensure_ascii=True, sort_keys=True) + "\n")
    return output_path


def training_record_to_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a stored Cognix training record into SFT messages."""
    task = str(record.get("task") or "")
    input_payload = record.get("input") if isinstance(record.get("input"), dict) else {}
    output_payload = record.get("output") if isinstance(record.get("output"), dict) else {}
    if task == "qa_citation":
        question = str(input_payload.get("question") or "").strip()
        answer = str(output_payload.get("answer") or "").strip()
        if not question or not answer:
            return []
        sources = input_payload.get("sources") if isinstance(input_payload.get("sources"), list) else []
        source_lines = []
        for source in sources[:8]:
            if not isinstance(source, dict):
                continue
            source_lines.append(
                f"- {source.get('source_path', 'unknown')}#chunk-{source.get('chunk_id', 'unknown')}: {source.get('excerpt', '')}"
            )
        context = "\n".join(source_lines)
        user_content = f"Question: {question}"
        if context:
            user_content += f"\n\nEvidence:\n{context}"
        return [
            {
                "role": "system",
                "content": "You are Cognix, a source-grounded knowledge assistant. Answer only from provided evidence and preserve citations.",
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
    return []
