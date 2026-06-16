from fastapi import APIRouter

from app.models.schemas import AskRequest, AskResponse, SourceSnippet
from app.services.llm import synthesize_answer
from app.services.outputs import save_analysis
from app.services.retrieval import retrieve

router = APIRouter(prefix="/api/ask", tags=["ask"])


@router.post("", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    chunks = retrieve(request.question)
    retrieval_summary = f"Retrieved {len(chunks)} chunks using local semantic and keyword search."
    answer = synthesize_answer(request.question, chunks, request.style)
    output_id = None
    output_path = None
    if request.save:
        output_id, path = save_analysis(request.question, answer, chunks, retrieval_summary)
        output_path = str(path)
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
        output_id=output_id,
        output_path=output_path,
    )
