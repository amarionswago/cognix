"""FastAPI routes for Cognix concept graph traversal."""

from fastapi import APIRouter

from app.db.graph import neighbors

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/neighbors/{slug}")
async def graph_neighbors(slug: str, depth: int = 1) -> dict:
    """Return neighboring graph concepts for a concept slug."""
    return {"concept": slug, "depth": depth, "neighbors": neighbors(slug.replace("-", " "), depth)}


@router.get("/concept/{slug}")
async def graph_concept(slug: str) -> dict:
    """Return a concept node and first-hop neighbors."""
    concept = slug.replace("-", " ")
    return {"concept": concept, "neighbors": neighbors(concept, 1)}
