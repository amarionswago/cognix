from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source: str = "manual"


class IngestResponse(BaseModel):
    job_id: int
    discovered: int
    processed: int
    skipped: int
    failed: int


class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    style: Literal["brief", "memo", "deep"] = "memo"
    save: bool = True


class SourceSnippet(BaseModel):
    chunk_id: int
    source_path: str
    excerpt: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceSnippet]
    retrieval_summary: str
    output_id: int | None = None
    output_path: str | None = None


class OutputUpdate(BaseModel):
    status: Literal["draft", "promoted", "archived", "deleted"]


class ProfileUpdate(BaseModel):
    username: str | None = None
    display_name: str = "Cognix User"
    theme: Literal["light", "dark"] = "light"
    default_answer_style: Literal["brief", "memo", "deep"] = "memo"
    raw_data_note: str = ""


class ProviderUpdate(BaseModel):
    enabled: bool = True
    api_key: str = ""
    model: str = ""


class HealthSummary(BaseModel):
    score: int
    totals: dict[str, int]
    findings: list[dict[str, Any]]
