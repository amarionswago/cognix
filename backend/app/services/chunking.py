from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    token_estimate: int


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)


def chunk_text(text: str, max_words: int = 450, overlap_words: int = 70) -> list[TextChunk]:
    words = text.split()
    if not words:
        return []
    chunks: list[TextChunk] = []
    start = 0
    index = 0
    step = max(1, max_words - overlap_words)
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(TextChunk(index=index, text=chunk, token_estimate=estimate_tokens(chunk)))
            index += 1
        if end == len(words):
            break
        start += step
    return chunks

