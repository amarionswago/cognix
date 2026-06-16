import json
from pathlib import Path


def adapt_known_export(path: Path, parsed_text: str) -> tuple[str, str] | None:
    lower_name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    if path.suffix.lower() == ".json" and ("ai_chats" in parts or "chatgpt" in lower_name):
        return adapt_chatgpt_json(path, parsed_text), "adapter-chatgpt-json"
    if path.suffix.lower() == ".html" and ("browser" in parts or "bookmarks" in lower_name):
        return parsed_text, "adapter-browser-html"
    if path.suffix.lower() == ".csv" and "finance" in parts:
        return parsed_text, "adapter-finance-csv"
    if path.suffix.lower() in {".csv", ".json", ".xml"} and "health" in parts:
        return parsed_text, "adapter-health-export"
    if path.suffix.lower() in {".eml", ".mbox"} or "email" in parts:
        return parsed_text, "adapter-email-basic"
    return None


def adapt_chatgpt_json(path: Path, parsed_text: str) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return parsed_text
    conversations = data if isinstance(data, list) else data.get("conversations", [])
    lines = [f"AI chat export: {path.name}"]
    for index, convo in enumerate(conversations[:200], start=1):
        if isinstance(convo, dict):
            title = convo.get("title") or convo.get("name") or f"Conversation {index}"
            lines.append(f"\n## {title}")
            mapping = convo.get("mapping")
            if isinstance(mapping, dict):
                for node in mapping.values():
                    message = node.get("message") if isinstance(node, dict) else None
                    if not message:
                        continue
                    author = message.get("author", {}).get("role", "unknown")
                    parts = message.get("content", {}).get("parts", [])
                    text = "\n".join(str(part) for part in parts if part)
                    if text.strip():
                        lines.append(f"{author}: {text.strip()}")
            elif "messages" in convo:
                for message in convo.get("messages", []):
                    role = message.get("role") or message.get("author") or "unknown"
                    content = message.get("content") or message.get("text") or ""
                    lines.append(f"{role}: {content}")
    return "\n".join(lines)
