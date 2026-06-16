from app.services.retrieval import RetrievedChunk
from app.services.providers import call_configured_model


def synthesize_answer(question: str, chunks: list[RetrievedChunk], style: str = "memo") -> str:
    """Generate a grounded answer.

    Version 1 intentionally uses deterministic synthesis so Cognix works without
    API keys. Cloud/local LLM providers can replace this through the same service.
    """
    if not chunks:
        return (
            f"# Answer\n\nI could not find enough indexed evidence to answer: {question}\n\n"
            "Add relevant files to `data/raw/`, run ingest, then ask again."
        )

    prompt = _build_grounded_prompt(question, chunks, style)
    provider_answer = call_configured_model(prompt)
    if provider_answer:
        return provider_answer.strip() + "\n"

    title = "Research Memo" if style != "brief" else "Answer"
    lines = [f"# {title}", "", f"**Question:** {question}", ""]
    if style == "brief":
        lines.extend(["## Short Answer", _brief_summary(chunks), ""])
    else:
        lines.extend(
            [
                "## Answer",
                "Cognix found relevant source material and prepared an evidence-backed draft. "
                "This version uses local deterministic synthesis; once LLM providers are configured, "
                "this service can produce deeper narrative analysis while preserving the same citations.",
                "",
                "## Evidence",
            ]
        )
        for index, chunk in enumerate(chunks, start=1):
            lines.append(f"{index}. {chunk.excerpt} `source: {chunk.source_path}, chunk: {chunk.chunk_id}`")
        lines.extend(["", "## Retrieval Notes", f"- Retrieved {len(chunks)} chunks.", "- Sources are ranked by semantic and keyword similarity."])
    return "\n".join(lines).strip() + "\n"


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
