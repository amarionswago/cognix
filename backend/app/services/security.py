import re
from pathlib import Path


SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(api[_-]?key|secret|password|token)\s*[:=]\s*[^\s]{8,}"),
]

def classify_source_type(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "ai_chats" in parts or "chatgpt" in parts or "claude" in parts or "codex" in parts:
        return "ai_interaction"
    if "code" in parts or ".git" in parts:
        return "code"
    if "finance" in parts:
        return "finance"
    if "health" in parts:
        return "health"
    if "browser" in parts:
        return "browser"
    if "media" in parts:
        return "media"
    if "education" in parts:
        return "education"
    if "creative" in parts:
        return "creative"
    if "location" in parts:
        return "location"
    if "email" in parts or "communication" in parts:
        return "communication"
    if "social" in parts:
        return "social"
    return "document"


def classify_sensitivity(path: Path, text_sample: str = "") -> str:
    return "research"


def has_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)
