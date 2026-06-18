import re
from functools import lru_cache

from app.config import get_settings
from app.services.cognix_micro_model import CognixMicroModel, load_cognix_micro_model
from app.services.sft_adapter import CognixSFTAdapter, load_sft_adapter
from app.services.retrieval import RetrievedChunk
from app.services.retrieval import detect_file_type_filter, keyword_group_hits, query_keyword_groups
from app.services.providers import call_configured_model


def synthesize_answer(question: str, chunks: list[RetrievedChunk], style: str = "memo") -> str:
    """Generate a grounded answer.

    Cognix can use a configured local trained micro-synthesis artifact, then a
    configured LLM provider, and finally deterministic synthesis as a no-key
    fallback.
    """
    if not chunks:
        return (
            f"# Answer\n\nI could not find enough indexed evidence to answer: {question}\n\n"
            "Add relevant files to `data/raw/`, run ingest, then ask again."
        )

    micro_model = load_optional_cognix_micro_model()
    if micro_model is not None:
        return micro_model.synthesize(question, chunks, style)
    sft_adapter = load_optional_sft_adapter()
    if sft_adapter is not None:
        return sft_adapter.synthesize(question, chunks, style)

    prompt = _build_grounded_prompt(question, chunks, style)
    provider_answer = call_configured_model(prompt)
    if provider_answer:
        return provider_answer.strip() + "\n"

    title = "Research Memo" if style != "brief" else "Answer"
    lines = [f"# {title}", "", f"**Question:** {question}", ""]
    if style == "brief":
        lines.extend(["## Short Answer", _extractive_summary(question, chunks, sentence_limit=2), ""])
    else:
        lines.extend(
            [
                "## Answer",
                _extractive_summary(question, chunks, sentence_limit=5),
                "",
                "## Evidence",
            ]
        )
        for index, chunk in enumerate(chunks, start=1):
            lines.append(f"{index}. {chunk.excerpt} `source: {chunk.source_path}, chunk: {chunk.chunk_id}`")
        lines.extend(["", "## Retrieval Notes", f"- Retrieved {len(chunks)} chunks.", "- Sources are ranked by semantic and keyword similarity."])
    return "\n".join(lines).strip() + "\n"


@lru_cache(maxsize=1)
def load_optional_cognix_micro_model() -> CognixMicroModel | None:
    """Load the trained local synthesis model only when explicitly selected."""
    settings = get_settings()
    backend = settings.synthesis_backend.strip().lower()
    if backend not in {"cognix-micro", "micro", "local-trained"}:
        return None
    path = settings.resolved_cognix_micro_model_path()
    if not path.exists():
        return None
    try:
        return load_cognix_micro_model(path)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_optional_sft_adapter() -> CognixSFTAdapter | None:
    """Load the trained SFT adapter only when explicitly selected."""
    settings = get_settings()
    backend = settings.synthesis_backend.strip().lower()
    if backend not in {"cognix-sft-adapter", "sft-adapter", "local-sft"}:
        return None
    path = settings.resolved_cognix_sft_adapter_path()
    if not path.exists():
        return None
    try:
        return load_sft_adapter(path)
    except Exception:
        return None


def _build_grounded_prompt(question: str, chunks: list[RetrievedChunk], style: str) -> str:
    source_lines = []
    for index, chunk in enumerate(chunks, start=1):
        source_lines.append(
            f"[{index}] source={chunk.source_path} chunk={chunk.chunk_id}\n{chunk.excerpt}"
        )
    return (
        "You are Cognix, a local-first research knowledge assistant. "
        "Answer only from the provided evidence. If evidence is insufficient, say so. "
        "Cite sources inline using source path and chunk id. "
        f"Answer style: {style}.\n\n"
        f"Question: {question}\n\n"
        "Evidence:\n"
        + "\n\n".join(source_lines)
    )


def _brief_summary(chunks: list[RetrievedChunk]) -> str:
    top = chunks[0]
    return f"The strongest indexed match is from `{top.source_path}` with score {top.score}. See chunk {top.chunk_id}."


def _extractive_summary(question: str, chunks: list[RetrievedChunk], sentence_limit: int) -> str:
    extension_filter = detect_file_type_filter(question)
    keyword_groups = query_keyword_groups(question, extension_filter)
    selected: list[str] = []

    for chunk in chunks:
        sentences = _sentences(chunk.excerpt)
        ranked = sorted(
            sentences,
            key=lambda sentence: keyword_group_hits(keyword_groups, sentence.lower()),
            reverse=True,
        )
        for sentence in ranked:
            if keyword_groups and keyword_group_hits(keyword_groups, sentence.lower()) == 0:
                continue
            selected.append(f"- {sentence} `source: {chunk.source_path}, chunk: {chunk.chunk_id}`")
            break
        if len(selected) >= sentence_limit:
            break

    if selected:
        return "\n".join(selected)

    top = chunks[0]
    return (
        "The retrieved evidence is related by source similarity, but it does not contain a clean direct statement "
        f"answering the question. Strongest source: `{top.source_path}`, chunk {top.chunk_id}."
    )


def _sentences(text: str) -> list[str]:
    compact = " ".join(text.split())
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", compact)]
    return [part for part in parts if len(part) > 40] or ([compact] if compact else [])
