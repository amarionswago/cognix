"""SQLite-backed concept graph utilities for Cognix v2."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.database import db_session, utc_now


@dataclass(frozen=True)
class GraphEdge:
    """A typed concept relationship stored in the lightweight graph layer."""

    source_concept: str
    target_concept: str
    relationship: str
    weight: float = 1.0
    source_file: str = ""
    source_chunk_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def upsert_graph_edge(edge: GraphEdge) -> int:
    """Insert a graph edge and return its database id."""
    now = utc_now()
    with db_session() as conn:
        cursor = conn.execute(
            """
            INSERT INTO graph_edges
            (source_concept, target_concept, relationship, weight, source_file,
             source_chunk_id, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalize_concept(edge.source_concept),
                normalize_concept(edge.target_concept),
                edge.relationship,
                edge.weight,
                edge.source_file,
                edge.source_chunk_id,
                json.dumps(edge.metadata, sort_keys=True),
                now,
                now,
            ),
        )
        return int(cursor.lastrowid)


def neighbors(concept: str, depth: int = 1) -> list[dict[str, Any]]:
    """Return graph neighbors up to a bounded traversal depth."""
    depth = max(1, min(depth, 3))
    normalized = normalize_concept(concept)
    with db_session() as conn:
        rows = conn.execute(
            """
            WITH RECURSIVE walk(node, target, relationship, weight, source_file, hop) AS (
                SELECT source_concept, target_concept, relationship, weight, source_file, 1
                FROM graph_edges
                WHERE source_concept = ?
                UNION ALL
                SELECT ge.source_concept, ge.target_concept, ge.relationship, ge.weight, ge.source_file, walk.hop + 1
                FROM graph_edges ge
                JOIN walk ON ge.source_concept = walk.target
                WHERE walk.hop < ?
            )
            SELECT node, target, relationship, weight, source_file, hop
            FROM walk
            ORDER BY hop, weight DESC, target
            """,
            (normalized, depth),
        ).fetchall()
    return [dict(row) for row in rows]


def normalize_concept(concept: str) -> str:
    """Normalize concept labels for stable graph keys."""
    return " ".join(concept.strip().lower().split())

