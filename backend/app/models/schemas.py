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
    ignored_removed: int = 0


class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    style: Literal["brief", "memo", "deep"] = "memo"
    save: bool = True


class SourceSnippet(BaseModel):
    chunk_id: int
    source_path: str
    excerpt: str
    score: float


class ConfidenceSnippet(BaseModel):
    score: float
    label: Literal["high", "medium", "low"]
    breakdown: dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceSnippet]
    retrieval_summary: str
    retrieval_diagnostics: dict[str, Any]
    confidence: ConfidenceSnippet
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


class IntelligenceRunRequest(BaseModel):
    use_llm: bool = False


class IntelligenceRunResponse(BaseModel):
    run_id: int
    status: str
    findings_created: int
    briefing_id: int
    briefing_path: str


class IntelligenceFindingResponse(BaseModel):
    id: int
    finding_type: str
    severity: str
    title: str
    description: str
    source_refs_json: str
    suggested_action: str
    status: str
    confidence: float
    metadata_json: str
    created_at: str
    updated_at: str
    resolved_at: str | None = None


class BriefingResponse(BaseModel):
    id: int
    brief_date: str
    title: str
    path: str
    summary: str
    finding_counts_json: str
    status: str
    content: str
    created_at: str
    updated_at: str


class MlCapabilityResponse(BaseModel):
    name: str
    state: Literal["ready", "configured", "fallback", "missing"]
    message: str
    detail: dict[str, Any]


class MlReadinessResponse(BaseModel):
    summary: dict[str, int]
    capabilities: list[MlCapabilityResponse]
