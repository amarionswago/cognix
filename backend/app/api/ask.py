from fastapi import APIRouter

from app.db.calibration import record_prediction, store_training_example
from app.db.confidence import EvidenceSource, compute_confidence, store_confidence_score
from app.models.schemas import AskRequest, AskResponse, ConfidenceSnippet, SourceSnippet
from app.services.llm import synthesize_answer
from app.services.outputs import save_analysis
from app.services.retrieval import retrieve_evidence

router = APIRouter(prefix="/api/ask", tags=["ask"])


@router.post("", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    evidence_pack = retrieve_evidence(request.question)
    chunks = evidence_pack.chunks
    retrieval_summary = (
        f"Retrieved {len(chunks)} chunks using {evidence_pack.retrieval_method}. "
        f"Subqueries: {', '.join(evidence_pack.subqueries)}. "
        f"Unique sources: {evidence_pack.diagnostics.get('unique_source_count', 0)}."
    )
    answer = synthesize_answer(request.question, chunks, request.style)
    confidence = compute_confidence(
        [EvidenceSource(chunk.source_path, chunk.score) for chunk in chunks]
    )
    output_id = None
    output_path = None
    if request.save:
        output_id, path = save_analysis(request.question, answer, chunks, retrieval_summary)
        output_path = str(path)
    store_confidence_score(output_id, request.question, confidence)
    record_prediction(
        "answer_confidence",
        {"question": request.question, "sources": [chunk.source_path for chunk in chunks]},
        confidence.label,
        confidence.score,
        "cognix-confidence-v1",
        confidence.breakdown,
    )
    if output_id:
        store_training_example(
            "qa_citation",
            {
                "question": request.question,
                "style": request.style,
                "sources": [
                    {"source_path": chunk.source_path, "chunk_id": chunk.chunk_id, "excerpt": chunk.excerpt}
                    for chunk in chunks
                ],
            },
            {"answer": answer, "confidence": confidence.score, "label": confidence.label},
            source=f"output:{output_id}",
        )
    return AskResponse(
        answer=answer,
        sources=[
            SourceSnippet(
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                excerpt=chunk.excerpt,
                score=chunk.score,
            )
            for chunk in chunks
        ],
        retrieval_summary=retrieval_summary,
        retrieval_diagnostics=evidence_pack.diagnostics,
        confidence=ConfidenceSnippet(
            score=confidence.score,
            label=confidence.label,
            breakdown=confidence.breakdown,
        ),
        output_id=output_id,
        output_path=output_path,
    )
