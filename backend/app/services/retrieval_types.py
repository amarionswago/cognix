"""Shared retrieval dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    source_path: str
    excerpt: str
    score: float
    sensitivity: str
